import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 128x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
        )

        # bottleneck
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(128 * 16 * 16, latent_dim)

        # reverse fully connected decoder part
        self.fc_dec = nn.Linear(latent_dim, 128 * 16 * 16)

        # decodre
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),                     # 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),                     # 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),                     # 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()   # because input is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc_enc(x)

        x = self.fc_dec(z)
        x = x.view(-1, 128, 16, 16)
        x = self.decoder(x)

        return x, z
