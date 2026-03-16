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
        self.d_k = canshu.get('d_k', 4)  # 动态获取 d_k（默认为 4）
        self.d_v = self.d_k

        # 定义 4 个独立的 LSTM 层
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=canshu[f'input_size_{i + 1}'],
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers)
                    # batch_first=True)  # batch_first=True 方便数据管理
            for i in range(4)
        ])

        # 定义 Query, Key, Value 层
        self.query_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.d_k) for _ in range(4)])
        self.key_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.d_k) for _ in range(4)])
        self.value_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.d_v) for _ in range(4)])

        # Dropout 层
        self.dropout = nn.Dropout(p=0.5)

        # 最终输出层
        self.out_layer = nn.Linear(self.d_v, 2)  # 输出 logits 而不是概率

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

    def attention(self, queries, keys, values):
        """
        改进后的注意力机制：使用矩阵运算
        """
        attn_weights = torch.bmm(queries, keys.transpose(1, 2))  # [batch_size, num_streams, num_streams]
        # print(attn_weights.shape,'sss')
        # exit()
        attn_weights = F.softmax(attn_weights, dim=-1)  # 归一化权重
        context = torch.bmm(attn_weights, values)  # [batch_size, num_streams, d_v]
        final_context = context.sum(dim=1)  # 汇总上下文
        return final_context
    
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
            lstm_out, (h_new, c_new) = lstm(x, (h, c))  # [seq_len, batch_size, hidden_size]
            lstm_outputs.append(lstm_out[-1, :, :])  # 获取最后一个时间步的输出 [batch_size, hidden_size]
            new_h_states.append(h_new)
            new_c_states.append(c_new)

        # 生成 Query, Key, Value
        queries = torch.stack([layer(out) for layer, out in zip(self.query_layers, lstm_outputs)], dim=1)  # [batch_size, 4, d_k]
        keys = torch.stack([layer(out) for layer, out in zip(self.key_layers, lstm_outputs)], dim=1)      # [batch_size, 4, d_k]
        values = torch.stack([layer(out) for layer, out in zip(self.value_layers, lstm_outputs)], dim=1)  # [batch_size, 4, d_v]

        # 注意力机制
        context = self.attention(queries, keys, values)  # [batch_size, d_v]

        # Dropout 和输出
        context = self.dropout(context)
        pred = self.out_layer(context)  # [batch_size, 2]
        # pred = F.softmax(pred, dim=1)  # 移除 softmax
        return pred, new_h_states, new_c_states
