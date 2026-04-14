import os

# ========================
# PARAMETERS
# ========================

canshu = {
    'batch_size_num': 128,
    'sequence_length_num': 18,
    'd_k': 1,
    'input_size_1': 20,
    'input_size_2': 8,
    'input_size_3': 23,
    'input_size_4': 4
}

model_name = "lstm"  # options: rnn, lstm


# ========================
# HYPERPARAMETERS
# ========================

lr = 0.1
hidden_size = 32      # Number of hidden neurons in RNN
num_layers = 1        # Number of RNN hidden layers

epoch_zong = 1000     # Total training epochs
epoch_show = 10       # Evaluation interval


# ========================

# ========================
# BASELINE
class Group1:
    x=2296
    y=2429
    z=x-500


# 17-3-3 to 17-9-3
class Group2:
    x = 388
    y = 540
    z = None


# 17-9-4 to 18-3-4
class Group3:
    x = 540
    y = 680
    z = x - 500


# 19-7-29 to 20-1-29
class Group4:
    x = 1049
    y = 1183
    z = x - 500


# 20-1-30 to 20-7-30
class Group5:
    x = 1184
    y = 1299
    z = x - 500


# 21-8-24 to 22-2-23
class Group6:
    x = 1599
    y = 1734
    z = x - 500


# 22-2-24 to 22-8-24
class Group7:
    x = 1734
    y = 1867
    z = x - 500