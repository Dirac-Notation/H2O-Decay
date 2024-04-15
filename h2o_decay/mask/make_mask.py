import torch
import matplotlib.pyplot as plt

# local = torch.load("local.pt").cpu().detach()
# h2o = torch.load("h2o.pt").cpu().detach()
# penalty2 = torch.load("penalty2.pt").cpu().detach()
# penalty8 = torch.load("penalty8.pt").cpu().detach()

# tensor_list = [local, h2o, penalty8, penalty2]

# plt.figure(figsize=(4*len(tensor_list), 4))

tensor_list = [torch.load("weights.pt").cpu().detach()]

for ln in range(32):
    for i,j in enumerate(tensor_list):
        tmp = j[ln].to(torch.float32)[:60,:60]

        # tmp *= -0.5
        # tmp += 0.5

        # ones = torch.ones_like(tmp)
        # ones = torch.triu(ones, diagonal=1)

        # tmp += ones*tmp

        plt.subplot(1, len(tensor_list), i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(1-tmp, cmap="gray")
        
    plt.savefig(f"scores/test_{ln}.png")