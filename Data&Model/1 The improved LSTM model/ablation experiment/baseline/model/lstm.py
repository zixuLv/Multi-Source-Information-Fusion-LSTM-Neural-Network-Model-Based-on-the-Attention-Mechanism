import torch  
import torch.nn as nn
import torch.nn.functional as F
import config as C

class Model1(nn.Module):
    def __init__(self, canshu, device):
        super(Model1, self).__init__()
        self.device = device
        self.hidden_size = C.hidden_size
        self.num_layers = C.num_layers

        # 定义单一的 LSTM 层
        self.lstm = nn.LSTM(
            input_size=canshu['input_size_1'],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=False  # 如果您的数据格式为 [batch, seq, feature]，将其设为 True
        )

        # Dropout 层
        self.dropout = nn.Dropout(p=0.5)

        # 最终输出层
        self.out_layer = nn.Linear(self.hidden_size, 2)  # 输出 logits

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
        """初始化隐藏状态和细胞状态"""
        # (num_layers, batch_size, hidden_size)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (h, c)

    def forward(self, x1 ,h, c):
        """
        前向传播

        参数:
            x1 (Tensor): 输入，形状为 [seq_len, batch_size, input_size]
            h (Tensor): 隐藏状态，形状为 [num_layers, batch_size, hidden_size]
            c (Tensor): 细胞状态，形状为 [num_layers, batch_size, hidden_size]

        返回:
            pred (Tensor): 预测结果，形状为 [batch_size, 2]
            h_new (Tensor): 更新后的隐藏状态
            c_new (Tensor): 更新后的细胞状态
        """
        # 通过 LSTM 层
        lstm_out, (h_new, c_new) = self.lstm(x1, (h, c))  # lstm_out: [seq_len, batch_size, hidden_size]

        # 获取最后一个时间步的输出
        last_output = lstm_out[-1, :, :]  # 形状为 [batch_size, hidden_size]

        # Dropout
        dropped = self.dropout(last_output)  # [batch_size, hidden_size]

        # 最终输出
        pred = self.out_layer(dropped)  # [batch_size, 2]
        # pred = F.softmax(pred, dim=1)  # 如果需要概率，可以保留；否则，输出 logits

        return pred, h_new, c_new
