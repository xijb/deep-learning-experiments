import torch

##入门
a = torch.arange(12)
print(a)
print(a.shape)
print(a.numel())
print(a.reshape(3,4))

b = torch.zeros(2,3,4)
c = torch.ones(2,3,4)
d = torch.randn(3,4)
print(b)
print(c)
print(d)

e = torch.tensor([[1,3,2,4],[5,6,8,1],[9,8,6,3]])
print(e)

x = torch.tensor([1,5,6,2])
y = torch.tensor([2,6,8,3])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
print(torch.exp(x))