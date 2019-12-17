# CMPUT651project

# preprocessing
preporcessing_final.py is for the data preprocessing.
The data is first read and split with useless symbols removed, then feed into sentiment tagging algorithm lib/MNegex.py. Then result data is dumped into Data/processed file.

The new dictionary is dumped into Data/Embedding/processed file.

Basically, the preprocessing rewrite sentiment word to "word1" for positive, and "word2" for negative. For example, a negated "bad" is rewritten as "bad1", while a non-negated "bad" is "bad2".

The new dictionary is built on current glove6B, with an additional all-zero dimension to store sentiments. Then, sentiment words such as "bad1" or "bad2" are added into the dictionary with sentiment dimension to be 1 or -1 while keep other features same as original Glove embedding.

# GRU
In order to train, it requires:
processed training file: Data/train_data.pkl
processed test data file: Data/test_data.pkl
processed word embedding file: Data/emb/glove.300d_sent.pkl or Data/emb/glove.200d_sent.pkl or Data/emb/glove.100d_sent.pkl
GRU model file: GRUNet.py
main trainig program: torch_GRU_gpu.py

The training parameters can be adjusted in file torch_GRU_gpu.py from line 42 to 94
The GPU to run can be changed at line 101

To start training, run "python3 torch_GRU_gpu.py" 
The program will produce outputs to stdout for every stage in the training
It will produce "acc.log" and "pred_y_[epoch].pkl" in "results" folder. 

# LSTMs
Standard_lstm_with_sentiment.py aims to use the features with embedded with Glove and sentiment information to build a standard LSTM model. Then this model is tested to get the final score which is the main evaluation measurement of the challenge, while standard_lstm_no_sentiment.py just uses the features with Glove embedding.
By simply changing nn.LSTM to nn.GRU in line 97 of  standard_lstm_with_sentiment.py or the corresponding line in standard_lstm_no_sentiment.py, standard GRU classification result can be derived.

What multi_layer_bilstm_with_sentiment.py is different from  standard_lstm_with_sentiment.py is the way the model is built.  What multi_layer_bilstm_with_sentiment.py is different from  multi_layer_bilstm_no_sentiment.py is also whether the features  embedded with sentiment information.
