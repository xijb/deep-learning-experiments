import torch
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6
x = multinomial.Multinomial(1 , fair_probs).sample()
y = multinomial.Multinomial(10 , fair_probs).sample()
counts = multinomial.Multinomial(1000 , fair_probs).sample() / 1000
print(x,y,counts)