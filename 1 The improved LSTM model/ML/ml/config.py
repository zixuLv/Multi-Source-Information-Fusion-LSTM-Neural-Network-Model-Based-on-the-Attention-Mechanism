# -*- coding: utf-8 -*- 
import os

beishu=1
# data_path=r'C:\Users\Administrator.USER-20210602UY\Desktop\加密货币\bitcoin2015-08-10..2021-12-17.csv'
model_list=["SupportVectorMachine","RandomForest","DecisionTree","Multi_LayerPerceptron","KNN","GBDT","Ada"]#"BayesClassifier",""QDA",Logistic_Regression","LinearDiscriminant"]

canshu = {'batch_size_num': 32, 
          'sequence_length_num': 18,
          'input_size_1': 4, 
          'input_size_2': 23,
          'input_size_3': 8, 
          'input_size_4': 20}