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

        # 定义 4 个独立的 LSTM 层
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=canshu[f'input_size_{i + 1}'],
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers
                    )  # 使用 batch_first=True 方便数据管理
            for i in range(4)
        ])

        # Dropout 层 
        self.dropout = nn.Dropout(p=0.5)

        # 最终输出层，输入维度为 4 * hidden_size（拼接后的特征维度）
        self.out_layer = nn.Linear(4 * self.hidden_size, 2)  # 输出 logits 而不是概率

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
        h_states = [torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) for _ in range(4)]
        c_states = [torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) for _ in range(4)]
        return h_states, c_states

    def forward(self, x1, x2, x3, x4, h_states, c_states):
        """
        前向传播
        """
        inputs = [x1, x2, x3, x4]
        lstm_outputs = []
        new_h_states = []
        new_c_states = []

        # 通过 LSTM 层获取输出
        for i, (x, lstm, h, c) in enumerate(zip(inputs, self.lstm_layers, h_states, c_states)):
            lstm_out, (h_new, c_new) = lstm(x, (h, c))  # [batch_size, seq_len, hidden_size]
            lstm_outputs.append(lstm_out[-1,:, :])  # 获取最后一个时间步的输出 [batch_size, hidden_size]
            new_h_states.append(h_new)
            new_c_states.append(c_new)

        # 拼接所有 LSTM 输出
        concatenated = torch.cat(lstm_outputs, dim=1)  # [batch_size, 4 * hidden_size]

        # Dropout 和输出
        context = self.dropout(concatenated)
        pred = self.out_layer(context)  # [batch_size, 2]
        # pred = F.softmax(pred, dim=1)  # 移除 softmax
        return pred, new_h_states, new_c_states
