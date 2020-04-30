
# if you want to use google's colab,
# you can upload compressed files to colab and uncompress them like below
# You can download them: http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html

# To use file_reader.py in colab, upload it to colab.
from file_reader import File_reader

fr = File_reader()

data, label = fr.load_ham_and_spam(
    ham_paths="default", spam_paths="default", max=3000)


import os
import pandas as pd
from collections import Counter

vocabs = [vocab for seq in data for vocab in seq.split()]
# a = [  word for seq in ["a d","b d","c d"] for word in seq.split() ]
# ['a', 'd', 'b', 'd', 'c', 'd']

vocab_count = Counter(vocabs)
# Count words in the whole dataset

#print(vocab_count)
# Counter({'the': 47430, 'to': 35684, 'and': 26245, 'of': 24176, 'a': 19290, 'in': 17442, 'you': 14258, ...

vocab_count = vocab_count.most_common(len(vocab_count))

vocab_to_int = {word : index+2 for index, (word, count) in enumerate(vocab_count)}
vocab_to_int.update({'__PADDING__': 0}) # index 0 for padding
vocab_to_int.update({'__UNKNOWN__': 1}) # index 1 for unknown word such as broken character

#print(vocab_to_int)
# {'the': 2, 'to': 3, 'and': 4, 'of': 5, 'a': 6, 'in': 7, 'you': 8, 'for': 9, "'": 10, 'is': 11, ...

import torch
from torch.autograd import Variable

# Tokenize & Vectorize sequences
vectorized_seqs = []
for seq in data:
  vectorized_seqs.append([vocab_to_int[word] for word in seq.split()])

# Save the lengths of sequences
seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))

# Add padding(0)
seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
  seq_tensor[idx, :seqlen] = torch.LongTensor(seq)


print(seq_lengths.max()) # tensor(30772)
print(seq_tensor[0]) # tensor([ 20,  77, 666,  ...,   0,   0,   0])
print(seq_lengths[0]) # tensor(412)


sample = "operations is digging out 2000 feet of pipe to begin the hydro test"

tokenized_sample = [ word for word in sample.split()]
print(tokenized_sample[:3]) # ['operations', 'is', 'digging']

vectorized_sample = [ vocab_to_int.get(word, 1) for word in tokenized_sample] # unknown word in dict marked as 1
print(vectorized_sample[:3]) # [424, 11, 14683]



import torch.utils.data.sampler as splr


class CustomDataLoader(object):
  def __init__(self, seq_tensor, seq_lengths, label_tensor, batch_size):
    self.batch_size = batch_size
    self.seq_tensor = seq_tensor
    self.seq_lengths = seq_lengths
    self.label_tensor = label_tensor
    self.sampler = splr.BatchSampler(splr.RandomSampler(self.label_tensor), self.batch_size, False)
    self.sampler_iter = iter(self.sampler)

  def __iter__(self):
    self.sampler_iter = iter(self.sampler) # reset sampler iterator
    return self

  def _next_index(self):
    return next(self.sampler_iter) # may raise StopIteration

  def __next__(self):
    index = self._next_index()

    subset_seq_tensor = self.seq_tensor[index]
    subset_seq_lengths = self.seq_lengths[index]
    subset_label_tensor = self.label_tensor[index]

    # order by length to use pack_padded_sequence()
    subset_seq_lengths, perm_idx = subset_seq_lengths.sort(0, descending=True)
    subset_seq_tensor = subset_seq_tensor[perm_idx]
    subset_label_tensor = subset_label_tensor[perm_idx]

    return subset_seq_tensor, subset_seq_lengths, subset_label_tensor

  def __len__(self):
    return len(self.sampler)


shuffled_idx = torch.randperm(label.shape[0])

seq_tensor = seq_tensor[shuffled_idx]
seq_lenghts = seq_lengths[shuffled_idx]
label = label[shuffled_idx]

PCT_TRAIN = 0.7
PCT_VALID = 0.2

length = len(label)
train_seq_tensor = seq_tensor[:int(length*PCT_TRAIN)]
train_seq_lengths = seq_lengths[:int(length*PCT_TRAIN)]
train_label = label[:int(length*PCT_TRAIN)]

valid_seq_tensor = seq_tensor[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))]
valid_seq_lengths = seq_lengths[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))]
valid_label = label[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))]

test_seq_tensor = seq_tensor[int(length*(PCT_TRAIN+PCT_VALID)):]
test_seq_lengths = seq_lengths[int(length*(PCT_TRAIN+PCT_VALID)):]
test_label = label[int(length*(PCT_TRAIN+PCT_VALID)):]

print(train_seq_tensor.shape) # torch.Size([4200, 30772])
print(valid_seq_tensor.shape) # torch.Size([1199, 30772])
print(test_seq_tensor.shape) # torch.Size([601, 30772])


# set shuffle = False since data is already shuffled
batch_size = 80
train_loader = CustomDataLoader(train_seq_tensor, train_seq_lengths, train_label, batch_size)
valid_loader = CustomDataLoader(valid_seq_tensor, valid_seq_lengths, valid_label, batch_size)
test_loader = CustomDataLoader(test_seq_tensor, test_seq_lengths, test_label, batch_size)


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SpamHamLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, n_layers,\
                 drop_lstm=0.1, drop_out = 0.1):

        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_lstm, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(drop_out)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, seq_lengths):

        # embeddings
        embedded_seq_tensor = self.embedding(x)

        # pack, remove pads
        packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

        # lstm
        packed_output, (ht, ct) = self.lstm(packed_input, None)
          # https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html
          # If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero

        # unpack, recover padded sequence
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # collect the last output in each batch
        last_idxs = (input_sizes - 1).to(device) # last_idxs = input_sizes - torch.ones_like(input_sizes)
        output = torch.gather(output, 1, last_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1, self.hidden_dim)).squeeze() # [batch_size, hidden_dim]

        # dropout and fully-connected layer
        output = self.dropout(output)
        output = self.fc(output).squeeze()

        # sigmoid function
        output = self.sig(output)

        return output



# Instantiate the model w/ hyperparams

vocab_size = len(vocab_to_int)
embedding_dim = 100 # int(vocab_size ** 0.25) # 15
hidden_dim = 15
output_size = 1
n_layers = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

net = SpamHamLSTM(vocab_size, embedding_dim, hidden_dim, output_size, n_layers, \
                 0.2, 0.2)
net = net.to(device)
print(net)


# loss and optimization functions
criterion = nn.BCELoss()

lr=0.03
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,\
                                                       mode = 'min', \
                                                      factor = 0.5,\
                                                      patience = 2)


import numpy as np

# training params

epochs = 6

counter = 0
print_every = 10
clip=5 # gradient clipping


net.train()
# train for some number of epochs
val_losses = []
for e in range(epochs):

    scheduler.step(e)

    for seq_tensor, seq_tensor_lengths, label in iter(train_loader):
        counter += 1

        seq_tensor = seq_tensor.to(device)
        seq_tensor_lengths = seq_tensor_lengths.to(device)
        label = label.to(device)

        # get the output from the model
        output = net(seq_tensor, seq_tensor_lengths)

        # get the loss and backprop
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()

        # prevent the exploding gradient
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss

            val_losses_in_itr = []
            sums = []
            sizes = []

            net.eval()

            for seq_tensor, seq_tensor_lengths, label in iter(valid_loader):

                seq_tensor = seq_tensor.to(device)
                seq_tensor_lengths = seq_tensor_lengths.to(device)
                label = label.to(device)
                output = net(seq_tensor, seq_tensor_lengths)

                # losses
                val_loss = criterion(output, label.float())
                val_losses_in_itr.append(val_loss.item())

                # accuracy
                binary_output = (output >= 0.5).short() # short(): torch.int16
                right_or_not = torch.eq(binary_output, label)
                sums.append(torch.sum(right_or_not).float().item())
                sizes.append(right_or_not.shape[0])

            accuracy = sum(sums) / sum(sizes)

            net.train()
            print("Epoch: {:2d}/{:2d}\t".format(e+1, epochs),
                  "Steps: {:3d}\t".format(counter),
                  "Loss: {:.6f}\t".format(loss.item()),
                  "Val Loss: {:.6f}\t".format(np.mean(val_losses_in_itr)),
                  "Accuracy: {:.3f}".format(accuracy))

# Epoch:  1/ 6	 Steps:  10	 Loss: 0.693371	 Val Loss: 0.689860	 Accuracy: 0.530
# Epoch:  1/ 6	 Steps:  20	 Loss: 0.699150	 Val Loss: 0.667903	 Accuracy: 0.585
# Epoch:  1/ 6	 Steps:  30	 Loss: 0.631709	 Val Loss: 0.626028	 Accuracy: 0.651
# Epoch:  1/ 6	 Steps:  40	 Loss: 0.609348	 Val Loss: 0.538908	 Accuracy: 0.716
# Epoch:  1/ 6	 Steps:  50	 Loss: 0.435395	 Val Loss: 0.440515	 Accuracy: 0.780
# Epoch:  2/ 6	 Steps:  60	 Loss: 0.364830	 Val Loss: 0.312334	 Accuracy: 0.892
# Epoch:  2/ 6	 Steps:  70	 Loss: 0.177650	 Val Loss: 0.283867	 Accuracy: 0.901
# Epoch:  2/ 6	 Steps:  80	 Loss: 0.379663	 Val Loss: 0.360904	 Accuracy: 0.883
# Epoch:  2/ 6	 Steps:  90	 Loss: 0.399583	 Val Loss: 0.390520	 Accuracy: 0.857
# Epoch:  2/ 6	 Steps: 100	 Loss: 0.467552	 Val Loss: 0.480415	 Accuracy: 0.808
# Epoch:  3/ 6	 Steps: 110	 Loss: 0.239100	 Val Loss: 0.282348	 Accuracy: 0.896
# Epoch:  3/ 6	 Steps: 120	 Loss: 0.091864	 Val Loss: 0.252968	 Accuracy: 0.915
# Epoch:  3/ 6	 Steps: 130	 Loss: 0.160094	 Val Loss: 0.209478	 Accuracy: 0.934
# I halted the training process at step 130

test_losses = []
sums = []
sizes = []

net.eval()

test_losses = []
for seq_tensor, seq_tensor_lengths, label in iter(test_loader):

    seq_tensor = seq_tensor.to(device)
    seq_tensor_lengths = seq_tensor_lengths.to(device)
    label = label.to(device)
    output = net(seq_tensor, seq_tensor_lengths)

    # losses
    test_loss = criterion(output, label.float())
    test_losses.append(test_loss.item())

    # accuracy
    binary_output = (output >= 0.5).short() # short(): torch.int16
    right_or_not = torch.eq(binary_output, label)
    sums.append(torch.sum(right_or_not).float().item())
    sizes.append(right_or_not.shape[0])

accuracy = np.sum(sums) / np.sum(sizes)
print("Test Loss: {:.6f}\t".format(np.mean(test_losses)),
      "Accuracy: {:.3f}".format(accuracy))

net.eval()
myString = "Have you been really busy this week? \
  Then you'll definitely want to make time for this lesson. \
  Have a wonderful week, learn something new, and practice some English!"


# get rid of some characters
unnecessary =  ["-", ".", ",", "/", ":", "@", "'", "!"]
content = myString.lower()
content = ''.join([c for c in content if c not in unnecessary])
input = [content]

# Tokenize & Vectorize sequences
vectorized_seqs = []
for seq in input:
  vectorized_seqs.append([vocab_to_int.get(word,1) for word in seq.split()])

# Save the lengths of sequences
seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))

# Add padding(0)
seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
  seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

# Predict
seq_tensor = seq_tensor.to(device)
seq_lengths = seq_lengths.to(device)
output = net(seq_tensor, seq_lengths)

print(output.item())
# 0.64 (>0.5), means SPAM (actually, it is a part of the advertisement of English lesson)



seqs = torch.tensor([[1,2,3,4,5], [6,7,8,0,0]])
lengths = torch.tensor([5,3], dtype = torch.int64).cpu()  # should be a 1D / CPU / int64 tensor
result = pack_padded_sequence(seqs, lengths, batch_first=True)
print(result.data) # tensor([1, 6, 2, 7, 3, 8, 4, 5])
print(result.batch_sizes) # tensor([2, 2, 2, 1, 1])

# seq_1) 1 2 3 4 5
# seq_2) 6 7 8 0 0
# batch) 2 2 2 1 1
