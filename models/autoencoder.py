import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # 128 to 64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 64 to 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 32 to 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # bottleneck
        self.flatten = nn.Flatten()
        self.bn_encoder = nn.Linear(128 * 16 * 16, latent_dim) # 128 channels x 16px x 16px
        self.bn_decoder = nn.Linear(latent_dim, 128 * 16 * 16) # latent vector

        self.decoder = nn.Sequential(
            # 16 to 32
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            # 32 to 64
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            # 64 to 128
            nn.ConvTranspose2d(
                32, 3,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid()  # output 0-1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.bn_encoder(x)

        x = self.bn_decoder(z)
        x = x.view(-1, 128, 16, 16)
        x = self.decoder(x)

        return x
