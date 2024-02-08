import torch.nn as nn
from torch_radon import Radon

from components import (
    ArtifactAffectedEncoder,
    ArtifactAffectedGenerator,
    Encoder,
    Generator,
)

class RadonADN(nn.Module):
    def __init__(self):
        super(RadonADN, self).__init__()
        self.Encoder = Encoder()
        self.ContentEncoder = ArtifactAffectedEncoder()
        self.ArtifactEncoder = ArtifactAffectedEncoder()

        self.ArtifactEncoder.down1.register_forward_hook(
            lambda module, input, output: setattr(self.ArtifactEncoder, 'x_artifact1', output.detach())
        )
        self.ArtifactEncoder.down2.register_forward_hook(
            lambda module, input, output: setattr(self.ArtifactEncoder, 'x_artifact2', output.detach())
        )

        self.Generator = Generator()
        self.ArtifactGenerator = ArtifactAffectedGenerator()

        self.Radon = Radon(
            resolution=256,
            angles=np.linspace(0, np.pi, 256, endpoint=False),
            clip_to_circle=False
        )

        self.SinoEncoder = Encoder()
        self.SinoContentEncoder = ArtifactAffectedEncoder()
        self.SinoArtifactEncoder = ArtifactAffectedEncoder()

        self.SinoArtifactEncoder.down1.register_forward_hook(
            lambda module, input, output: setattr(self.SinoArtifactEncoder, 'x_artifact1', output.detach())
        )
        self.SinoArtifactEncoder.down2.register_forward_hook(
            lambda module, input, output: setattr(self.SinoArtifactEncoder, 'x_artifact2', output.detach())
        )

    def forward(self, x, y):
        x_content = self.ContentEncoder(x) + self.SinoContentEncoder(self.Radon.forward(x))
        x_denoised = self.Generator(x_content)

        x_artifact3 = self.ArtifactEncoder(x) + self.SinoArtifactEncoder(self.Radon.forward(x))
        self.ArtifactEncoder.x_artifact2 = self.ArtifactEncoder.x_artifact2 + self.SinoArtifactEncoder.x_artifact2
        self.ArtifactEncoder.x_artifact1 = self.ArtifactEncoder.x_artifact1 + self.SinoArtifactEncoder.x_artifact1

        x_reconstructed = self.ArtifactGenerator(
            x_content,
            x_artifact3,
            self.ArtifactEncoder.x_artifact2,
            self.ArtifactEncoder.x_artifact1,
        )

        y_content = self.Encoder(y) + self.SinoEncoder(self.Radon.forward(x))

        y_noisy = self.ArtifactGenerator(
            y_content,
            x_artifact3,
            self.ArtifactEncoder.x_artifact2,
            self.ArtifactEncoder.x_artifact1,
        )

        y_denoised = self.Generator(self.ContentEncoder(y_noisy) + self.SinoContentEncoder(self.Radon.forward(y_noisy)))
        y_reconstructed = self.Generator(y_content)

        return x_denoised, x_reconstructed, y_noisy, y_denoised, y_reconstructed

    def simple_forward(self, x):
        x_content = self.ContentEncoder(x)
        x_denoised = self.Generator(x_content)

        return x_denoised
