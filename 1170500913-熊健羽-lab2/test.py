import torch
from train import test
# AlexNet1 = torch.load(f"AlexNet1.ckpt")
# print("AlexNet1:")
# test(AlexNet1)

AlexNet2 = torch.load(f"AlexNet2.ckpt")  # best model
print("AlexNet2:")
test(AlexNet2)

# AlexNet3 = torch.load(f"AlexNet3.ckpt")
# print("AlexNet3:")
# test(AlexNet3)

# AlexNet4 = torch.load(f"AlexNet4.ckpt")
# print("AlexNet4:")
# test(AlexNet4)


