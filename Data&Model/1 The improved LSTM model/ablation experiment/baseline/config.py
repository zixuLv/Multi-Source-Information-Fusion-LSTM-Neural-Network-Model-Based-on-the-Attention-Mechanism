import os

# -----------------------------
# Configuration parameters
# -----------------------------

canshu = {
    'batch_size_num': 128,
    'sequence_length_num': 18,
    'd_k': 1,
    'input_size_1': 20,
    'input_size_2': 8,
    'input_size_3': 23,
    'input_size_4': 4
}

model_name = "lstm"  # Options: "rnn", "lstm"

# -----------------------------
# Hyperparameters
# -----------------------------

lr = 0.1  # Learning rate

hidden_size = 32      # Number of hidden units in RNN
num_layers = 1        # Number of RNN layers

epoch_zong = 1000     # Total training epochs
epoch_show = 10       # Interval for displaying training results