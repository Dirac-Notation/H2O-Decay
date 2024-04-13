import torch
import matplotlib.pyplot as plt

h2o = torch.load("h2o.pt").cpu().detach()
penalty = torch.load("penalty.pt").cpu().detach()

tmp = penalty[0].to(torch.float32)[:30,:30]

tmp *= -0.5
tmp += 0.5

ones = torch.ones_like(tmp)
ones = torch.triu(ones, diagonal=1)

tmp += ones*tmp

plt.xticks([])
plt.yticks([])
plt.imshow(tmp, cmap="gray")
plt.grid(True)
plt.savefig("test.png")