import itertools
import numpy as np

ngram_range = [(1,1),(1,2),(2,2),(1,3),(2,3),(3,3),(1,4),(2,4),(3,4),(4,4)]
max_df = np.linspace(0,1,11)
min_df = np.linspace(0,1,11)

for xs in itertools.product(ngram_range, max_df, min_df):
    print(xs)







bestComb = {}
accuracyBefore = 0.0
#Structure
for comb in combinations:
    train()
    acc()
    if acc() > accuracyBefore:
        bestComb = comb
