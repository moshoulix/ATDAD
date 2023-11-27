"""
16        32        73
E_r(x) ←--- E_r ←--- x
        ↘
z ----→ G1 -→ G2 -→ x_rec
                  ↘
z_rec ← E_2 ← E_1 ← G(z)

D_x: x ←--→ x_rec
D_y: y ←--→ y_rec
D_z: z ←--→ z_rec
D_adv: (z, G(z)) ←--→ (E_r(x), x)
"""
import torch.nn as nn
import torch


def gen_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding),
        nn.InstanceNorm1d(out_channels, affine=True),
        nn.LeakyReLU(0.2, True)
    )


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # N x 1 x 16
            gen_block(1, 64, 4, 1, 0),
            # N x 64 x 19
            gen_block(64, 16, 4, 2, 1),
            # N x 16 x 38
            gen_block(16, 4, 4, 2, 1),
            # N x 4 x 76
            nn.Flatten(),
            # N x 304
            nn.Linear(304, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 73),
            # N x 73
            nn.Tanh()
        )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs.reshape(-1, 1, 73)


class DiscriminatorA(nn.Module):
    def __init__(self):
        super(DiscriminatorA, self).__init__()
        self.model = nn.Sequential(
            # N x 1 x 73
            nn.Conv1d(1, 64, kernel_size=6, padding=2),
            # N x 64 x 72
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(3, stride=2),
            # N x 64 x 35
            nn.Conv1d(64, 64, kernel_size=6, padding=2),
            # N x 64 x 34
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(3, stride=2),
            # N x 64 x 16
            nn.Flatten(),
            nn.Linear(64 * 16, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, inputs):
        return self.model(inputs)


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        self.inf_z = nn.Sequential(
            # N x 1 x 16
            nn.Conv1d(1, 32, kernel_size=6, padding=2),
            # N x 32 x 15
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(3, stride=2),
            # N x 32 x 7
        )

        self.inf_z_rec = nn.Sequential(
            # N x 1 x 16
            nn.Conv1d(1, 32, kernel_size=6, padding=2),
            # N x 32 x 15
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(3, stride=2),
            # N x 32 x 7
        )

        self.model = nn.Sequential(
            # N x 32 x 14
            nn.Conv1d(32, 16, kernel_size=4, padding=1, stride=2),
            # N x 16 x 7
            nn.Flatten(),
            nn.Linear(112, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 1)
        )

    def forward(self, z, rec):
        inputs = torch.cat((self.inf_z(z), self.inf_z_rec(rec)), dim=2)
        return self.model(inputs)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            # N x 1 x 73
            nn.Conv1d(1, 64, kernel_size=4, padding=2),
            # N x 64 x 74
            nn.LeakyReLU(0.2, True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            # N x 64 x 36
            nn.Conv1d(64, 16, kernel_size=4, stride=2, padding=1),
            # N x 16 x 18
            nn.LeakyReLU(0.2, True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            # N x 16 x 8
            nn.Flatten(),
            # N x 128
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 16)
            # N x 16

        )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs.reshape(-1, 1, 16)


class DiscriminatorX(nn.Module):
    def __init__(self):
        super(DiscriminatorX, self).__init__()
        self.inf_x = nn.Sequential(
            # N x 1 x 73
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            # N x 32 x 36
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(3, stride=2),
            # N x 32 x 17
        )

        self.inf_x_rec = nn.Sequential(
            # N x 1 x 73
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            # N x 32 x 36
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(3, stride=2),
            # N x 32 x 17
        )

        self.model = nn.Sequential(
            # N x 32 x 34
            nn.Conv1d(32, 8, kernel_size=4, stride=2, padding=1),
            # N x 8 x 17
            nn.Flatten(),
            nn.Linear(8 * 17, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 1)
        )

    def forward(self, x, rec):
        inputs = torch.cat((self.inf_x(x), self.inf_x_rec(rec)), dim=2)
        return self.model(inputs)
