preporcessing_final.py is for the data preprocessing.
The data is first read and split with useless symbols removed, then feed into sentiment tagging algorithm lib/MNegex.py. Then result data is dumped into Data/processed file.

The new dictionary is dumped into Data/Embedding/processed file.

Basically, the preprocessing rewrite sentiment word to "word1" for positive, and "word2" for negative. For example, a negated "bad" is rewritten as "bad1", while a non-negated "bad" is "bad2".

The new dictionary is built on current glove6B, with an additional all-zero dimension to store sentiments. Then, sentiment words such as "bad1" or "bad2" are added into the dictionary with sentiment dimension to be 1 or -1 while keep other features same as original Glove embedding.
