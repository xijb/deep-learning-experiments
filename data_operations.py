import torch

# ##入门
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

f = torch.arange(12, dtype=torch.float32).reshape((3,4))
g = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((f,g) , dim=0))
print(torch.cat((f,g) , dim=1))
print(f == g)
print(f.sum())

h = torch.arange(3).reshape((3,1))
i = torch.arange(2).reshape((1,2))
print(h)
print(i)
print(h + i)