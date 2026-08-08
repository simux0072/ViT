import torch


class Vit(torch.nn.Module):
    def __init__(
        self,
        imageShape: list[int],
        batchSize: int,
        patchSize: list[int],
        embedDims: int = 512,
    ) -> None:
        super().__init__()

        self.imageShape: list[int] = imageShape
        self.batchSize: int = batchSize
        self.patchSize: list[int] = patchSize
        self.embedDims: int = embedDims

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

    def createEmbeddings(self, images: torch.Tensor) -> torch.Tensor:
        if (
            images.shape[-1] % self.patchSize[-1] != 0
            or images.shape[-2] % self.patchSize[-2] != 0
        ):
            raise ValueError(
                f"The image shape: {images.shape} is not divisible by the patch shape: {self.patchSize}."
            )

        embeddings: torch.Tensor = self.patchEmbeddings(images)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        clsToken = self.clsToken.expand(self.batchSize, -1, -1)
        embeddings = torch.concat((clsToken, embeddings), dim=1)

        fullEmbeddings = embeddings + self.posEmbeddings

        return fullEmbeddings
