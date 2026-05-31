import torch
import torch.nn as nn
import config as C
import csv
import numpy as np
class model1(nn.Module):
    def __init__(self,canshu):
        super().__init__()
        # print(canshu)
        self.batch_size_num = canshu['batch_size_num']
        self.sequence_length_num = canshu['sequence_length_num']
        self.input_size_1 = canshu['input_size_1']
        self.input_size_2 = canshu['input_size_2']
        self.input_size_3 = canshu['input_size_3']
        self.input_size_4 = canshu['input_size_4']
        
        self.rnn_1 = nn.RNN(
            input_size=C.canshu['input_size_1'],
            hidden_size=C.hidden_size,  # RNN隐藏神经元个数
            num_layers=C.num_layers,  # RNN隐藏层个数
        )
        
        self.rnn_2 = nn.RNN(
            input_size=C.canshu['input_size_2'],
            hidden_size=C.hidden_size,  # RNN隐藏神经元个数
            num_layers=C.num_layers,  # RNN隐藏层个数
        )
        
        self.rnn_3 = nn.RNN(
            input_size=C.canshu['input_size_3'],
            hidden_size=C.hidden_size,  # RNN隐藏神经元个数
            num_layers=C.num_layers,  # RNN隐藏层个数
        )
        
        self.rnn_4 = nn.RNN(
            input_size=C.canshu['input_size_4'],
            hidden_size=C.hidden_size,  # RNN隐藏神经元个数
            num_layers=C.num_layers,  # RNN隐藏层个数
        )
        self.out_1_2=nn.Sequential(nn.Linear(C.canshu['sequence_length_num'],1),
                               nn.Sigmoid(),
                                )
        self.out_2_2=nn.Sequential(nn.Linear(C.canshu['sequence_length_num'],1),
                               nn.Sigmoid(),
                                )
        self.out_3_2=nn.Sequential(nn.Linear(C.canshu['sequence_length_num'],1),
                               nn.Sigmoid(),
                                )
        self.out_4_2=nn.Sequential(nn.Linear(C.canshu['sequence_length_num'],1),
                                nn.Sigmoid(),
        )
        
        #混合第一次
        self.q1 = nn.Sequential(nn.Linear(C.hidden_size,64),
                                )
        self.k1 = nn.Sequential(nn.Linear(C.hidden_size,64),
                                )
        
        self.q2 = nn.Sequential(nn.Linear(C.hidden_size,64),
                                )
        self.k2 = nn.Sequential(nn.Linear(C.hidden_size,64),
                                )
                                
        self.q3 = nn.Sequential(nn.Linear(C.hidden_size,64),
                                )
        self.k3 = nn.Sequential(nn.Linear(C.hidden_size,64),
                                )
                                
        self.q4 = nn.Sequential(nn.Linear(C.hidden_size,64),
                                )
        self.k4 = nn.Sequential(nn.Linear(C.hidden_size,64),
                                )
        #混合第二次
        self.qq1 = nn.Sequential(nn.Linear(64,32),
                                )
        self.kk1 = nn.Sequential(nn.Linear(64,32),
                                )
        
        self.qq2 = nn.Sequential(nn.Linear(64,32),
                                )
        self.kk2 = nn.Sequential(nn.Linear(64,32),
                                )
        
        self.out_zuihou = nn.Sequential(nn.Linear(32,2)
                                    ,nn.Sigmoid())
        
        
        self.cahnshi1 = nn.Sequential(nn.Linear(64,2)
                                    ,nn.Sigmoid())
        self.cahnshi2 = nn.Sequential(nn.Linear(64,2)
                                    ,nn.Sigmoid())
        self.cahnshi3 = nn.Sequential(nn.Linear(64,2)
                                    ,nn.Sigmoid())
        self.cahnshi4 = nn.Sequential(nn.Linear(64,2)
                                    ,nn.Sigmoid())
        
        
    def forward(self, x1, x2 ,x3, x4,(c1,h1),(c2,h2),(c3,h3),(c4,h4)):
        # print(x1.shape,"===========",x2.shape,"===========",x3.shape,"===========",x4.shape)
        # 做rnn
        out_1_1, h_state1 = self.rnn_1(x1,h_state1)
        out_2_1, h_state2 = self.rnn_2(x2,h_state2)
        out_3_1, h_state3 = self.rnn_3(x3,h_state3)
        out_4_1, h_state4 = self.rnn_4(x4,h_state4)
        # print(out_1_1[0][0],"++++",out_2_1[0][0],"++++",out_3_1[0][0],"++++",out_4_1[0][0])
        #把形状转回来
        out_1_1 = out_1_1.view(self.batch_size_num,-1,self.sequence_length_num)
        out_2_1 = out_2_1.view(self.batch_size_num,-1,self.sequence_length_num)
        out_3_1 = out_3_1.view(self.batch_size_num,-1,self.sequence_length_num)
        out_4_1 = out_4_1.view(self.batch_size_num,-1,self.sequence_length_num)
        # print(out_1_1.shape,"=============",out_2_1.shape,"=============",out_3_1.shape,"=============",out_4_1.shape)
        #合并为一列
        out_1_2 = np.squeeze(self.out_1_2(out_1_1),-1)
        out_2_2 = np.squeeze(self.out_2_2(out_2_1),-1)
        out_3_2 = np.squeeze(self.out_3_2(out_3_1),-1)
        out_4_2 = np.squeeze(self.out_4_2(out_4_1),-1)
        # print(out_1_2.shape,"++++",out_2_2.shape,"++++",out_3_2.shape,"++++",out_4_2.shape)
        # exit()
        #开始做注意力机制
        #print(out_1_2.shape)#,"=============",out_2_2.shape,"=============",out_3_2.shape,"=============",out_4_2.shape)
        #exit()
        # out_1_2_q = self.q1(out_1_2)
        # out_1_2_k = self.k1(out_1_2)
        # print(out_1_2_q[0],"=======",out_1_2_k[0])
        # exit()
        #第二条线
        # out_2_2_q = self.q2(out_2_2)
        # out_2_2_k = self.k2(out_2_2)
        #第三条线
        # out_3_2_q = self.q3(out_3_2)
        # out_3_2_k = self.k3(out_3_2)
        #第四条线
        # out_4_2_q = self.q4(out_4_2)
        # out_4_2_k = self.k4(out_4_2)
        # print(out_4_2_q[0],"=======",out_4_2_k[0])
        
        #开始混合
        # hun_1_q = out_1_2_q*out_2_2_q
        # hun_1_k = out_1_2_k*out_2_2_k
        # hun_1_2 = hun_1_q + hun_1_k
        # print(hun_1_2.shape,"=============",hun_1_k.shape)
        
        # hun_2_q = out_3_2_q*out_4_2_q
        # hun_2_k = out_3_2_k*out_4_2_k
        # hun_3_4 = hun_2_q + hun_2_k
        # print(hun_3_4.shape,"=============",hun_2_q.shape)
        
        #第二次混，第一条线
        # outout_1_q = self.qq1(hun_1_2)
        # outout_1_k = self.kk1(hun_1_2)
        #第二次混，第二条线
        # outout_2_q = self.qq2(hun_1_2)
        # outout_2_k = self.kk2(hun_1_2)
        #开始第二次混合
        # hunhun_1_q = outout_1_q*outout_2_q
        # hunhun_1_k = outout_1_k*outout_2_k
        # hunhun_1_2 = hunhun_1_q + hunhun_1_k
        
        
        # out = out_1_2*out_2_2*out_3_2*out_4_2
        # out = self.out_zuihou(hunhun_1_2)
        
        out_1_2 = self.cahnshi1(out_1_2)
        out_2_2 = self.cahnshi1(out_2_2)
        out_3_2 = self.cahnshi1(out_3_2)
        out_4_2 = self.cahnshi1(out_4_2)
        
        out = out_1_2*out_2_2*out_3_2*out_4_2
        return out,h_state1,h_state2,h_state3,h_state4
    
  