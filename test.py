import torch

import vit

vitClass = vit.Vit([3, 256, 256], 1, [16, 16])

print(vitClass.createEmbeddings(torch.zeros([1, 3, 256, 256])).shape)
