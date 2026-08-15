import torch

import vit

vitClass = vit.Vit([3, 256, 256], 2, [16, 16], 512, 4)
images = torch.randn((2, 3, 256, 256))
images2 = torch.randn((2, 3, 256, 256))

embeddings = vitClass.createEmbeddings(images)
embeddings = vitClass.attentionNorm(embeddings)
attention = vitClass.multiHeadAttention(embeddings)
attention = vitClass.ffNorm(attention)
ffn = vitClass.feedForward(attention)

print(ffn.shape)

ffn = vitClass(images, True)
print(images2.shape)
