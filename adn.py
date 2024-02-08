import torch.nn as nn

from components import (
    ArtifactAffectedEncoder,
    ArtifactAffectedGenerator,
    Encoder,
    Generator,
)

class ADN(nn.Module):
    def __init__(self):
        super(ADN, self).__init__()
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

    def forward(self, x, y):
        x_content = self.ContentEncoder(x)
        x_denoised = self.Generator(x_content)
        x_artifact3 = self.ArtifactEncoder(x)
        x_reconstructed = self.ArtifactGenerator(
            x_content,
            x_artifact3,
            self.ArtifactEncoder.x_artifact2,
            self.ArtifactEncoder.x_artifact1,
        )

        y_content = self.Encoder(y)
        y_noisy = self.ArtifactGenerator(
            y_content,
            x_artifact3,
            self.ArtifactEncoder.x_artifact2,
            self.ArtifactEncoder.x_artifact1,
        )
        y_denoised = self.Generator(self.ContentEncoder(y_noisy))
        y_reconstructed = self.Generator(y_content)

        return x_denoised, x_reconstructed, y_noisy, y_denoised, y_reconstructed

    def simple_forward(self, x):
        x_content = self.ContentEncoder(x)
        x_denoised = self.Generator(x_content)

        return x_denoised
