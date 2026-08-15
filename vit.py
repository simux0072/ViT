import math

import torch


class Vit(torch.nn.Module):
    def __init__(
        self,
        imageShape: list[int],
        batchSize: int,
        patchSize: list[int],
        embedDims: int = 512,
        nHeads: int = 4,
    ) -> None:
        super().__init__()

        self.imageShape: list[int] = imageShape
        self.batchSize: int = batchSize
        self.patchSize: list[int] = patchSize
        self.embedDims: int = embedDims

        if embedDims % nHeads != 0:
            raise ValueError(
                f"Embedding Dimension {embedDims} is not divisible by Number of Heads {nHeads}."
            )
        self.nHeads: int = nHeads
        self.headDims: int = int(embedDims / nHeads)

        self.patchNum = int(
            self.imageShape[-1]
            / self.patchSize[-1]
            * self.imageShape[-2]
            / self.patchSize[-2]
        )

        self.patchEmbeddings = torch.nn.Conv2d(
            in_channels=3,
            out_channels=embedDims,
            kernel_size=(self.patchSize[-2], self.patchSize[-1]),
            stride=(self.patchSize[-2], self.patchSize[-1]),
        )

        self.posEmbeddings = torch.nn.Parameter(
            torch.rand((self.batchSize, self.patchNum + 1, self.embedDims))
        )
        self.clsToken = torch.nn.Parameter(torch.rand((1, 1, self.embedDims)))

        self.attentionNorm = torch.nn.LayerNorm(self.embedDims)

        self.queryLayer = torch.nn.Linear(
            in_features=self.embedDims, out_features=self.embedDims
        )

        self.keyLayer = torch.nn.Linear(
            in_features=self.embedDims, out_features=self.embedDims
        )

        self.valueLayer = torch.nn.Linear(
            in_features=self.embedDims, out_features=self.embedDims
        )

        self.attentionOutput = torch.nn.Linear(
            in_features=self.embedDims, out_features=self.embedDims
        )

        self.ffNorm = torch.nn.LayerNorm(self.embedDims)

        self.linear1 = torch.nn.Linear(
            in_features=self.embedDims, out_features=4 * self.embedDims
        )
        self.linear2 = torch.nn.Linear(
            in_features=4 * self.embedDims, out_features=self.embedDims
        )

    def createEmbeddings(self, images: torch.Tensor) -> torch.Tensor:

        if (images.shape[-1] % self.patchSize[-1] != 0) or images.shape[
            -2
        ] % self.patchSize[-2] != 0:
            raise ValueError(
                f"The image shape: {images.shape} is not divisible by the patch shape: {self.patchSize}."
            )

        embeddings: torch.Tensor = self.patchEmbeddings(images)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        clsToken = self.clsToken.expand(self.batchSize, -1, -1)
        embeddings = torch.concat((clsToken, embeddings), dim=1)

        fullEmbeddings = embeddings + self.posEmbeddings

        return fullEmbeddings

    def multiHeadAttention(self, embeddings: torch.Tensor) -> torch.Tensor:

        queryTensor: torch.Tensor = self.queryLayer(embeddings).view(
            self.batchSize, -1, embeddings.shape[1], self.headDims
        )
        keyTensor: torch.Tensor = (
            self.keyLayer(embeddings)
            .view(self.batchSize, -1, embeddings.shape[1], self.headDims)
            .transpose(-2, -1)
        )
        valueTensor: torch.Tensor = self.valueLayer(embeddings).view(
            self.batchSize, -1, embeddings.shape[1], self.headDims
        )

        attention: torch.Tensor = (
            torch.nn.functional.softmax(
                (queryTensor @ keyTensor) / math.sqrt(self.nHeads), dim=-1
            )
            @ valueTensor
        )

        embeddings = (
            self.attentionOutput(attention.transpose(1, 2).flatten(-2, -1)) + embeddings
        )

        return embeddings

    def feedForward(self, embeddings: torch.Tensor) -> torch.Tensor:
        feedForward = self.linear1(embeddings)
        feedForward = torch.nn.functional.gelu(feedForward)
        feedForward = self.linear2(feedForward) + embeddings
        return feedForward

    def forward(self, input: torch.Tensor, patchingNeeded: bool):
        if patchingNeeded:
            embeddings = self.createEmbeddings(input)

        embeddings = self.attentionNorm(embeddings)

        attention = self.multiHeadAttention(embeddings)
        embeddings = self.ffNorm(attention)

        embeddings = self.feedForward(embeddings)
        return embeddings
