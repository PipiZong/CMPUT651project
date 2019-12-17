The program requires Pytorch of version 1.3.1 and cuda

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