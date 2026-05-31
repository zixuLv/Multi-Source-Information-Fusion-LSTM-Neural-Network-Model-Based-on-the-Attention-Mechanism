import os
#CANSHU

canshu = {'batch_size_num': 128, 
          'sequence_length_num': 18,
          'd_k': 1,
          'input_size_1': 20, 
          'input_size_2': 8,
          'input_size_3': 23, 
          'input_size_4': 4}
model_name = "lstm"  #rnn,lstm

#超参数
lr = 0.1
hidden_size = 32 # RNN隐藏神经元个数
num_layers = 1  # RNN隐藏层个数



epoch_zong = 2000#训练轮数
epoch_show = 1
