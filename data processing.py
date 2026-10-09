import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]        ##分别赋值，索引，切片
inputs = pd.get_dummies(inputs, dummy_na=True)              ##做二分类
inputs = inputs.fillna(inputs.mean())                       ##用均值填补NaN
print(inputs)

x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(x)
print(y)

####   Linear Algebra
A = torch.arange(20, dtype=torch.float32).reshape((5,4))                            ##初始化矩阵
print(A)
print(A.T)                                                     ##矩阵的转置
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])            ##对称矩阵
print(B == B.T)

X = torch.arange(20, dtype=torch.float32).reshape((5,4))
Y = X.clone()
print(Y)
print(X * Y)                                                   ## Hadamard积（Hadamard product）
print(X.shape)

A_sum_axis0 = A.sum(axis=0)
A_sum_axis1 = A.sum(axis=1)
A_sum = A.sum(axis=[0,1])
A_mean = A.mean(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)
print(A_sum_axis1)
print(A_sum)
print(A_sum.shape)
print(A_mean)