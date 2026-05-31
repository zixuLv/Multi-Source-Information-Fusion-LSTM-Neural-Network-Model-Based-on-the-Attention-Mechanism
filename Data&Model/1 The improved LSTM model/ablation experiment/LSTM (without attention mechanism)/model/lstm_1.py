import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C

class Model1(nn.Module):
    def __init__(self, canshu, device):
        super().__init__()
        self.device = device
        self.hidden_size = C.hidden_size
        self.num_layers = C.num_layers

        # 定义单个 LSTM 层
        self.lstm = nn.LSTM(input_size=canshu['input_size_1'],
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers)
                            # batch_first=True)  # 如果需要，可以设置 batch_first=True

        # Dropout 层
        self.dropout = nn.Dropout(p=0.5)

        # 最终输出层
        self.out_layer = nn.Linear(self.hidden_size, 2)  # 输出 logits 而不是概率

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def init_hidden(self, batch_size):
        """动态初始化隐藏状态"""
        h_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h_state, c_state

    def forward(self, x, h_state, c_state):
        """
        前向传播
        """
        lstm_out, (h_new, c_new) = self.lstm(x, (h_state, c_state))  # [seq_len, batch_size, hidden_size]
        lstm_last_output = lstm_out[-1, :, :]  # 获取最后一个时间步的输出 [batch_size, hidden_size]

        # Dropout 和输出
        context = self.dropout(lstm_last_output)
        pred = self.out_layer(context)  # [batch_size, num_classes]
        return pred, h_new, c_new
