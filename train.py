import pandas as pd

from nets import Generator, Encoder, DiscriminatorA, DiscriminatorZ, DiscriminatorX
from dataset import ids_dataloader
from utils.plot import show_density, show_flow
from utils.seed import set_seed

import torch
import torch.nn as nn
from torch import optim
from barbar import Bar


set_seed(1234)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.dataloader = toy_dataloader(
            npz=args.dataset_dir, name=args.dataset_name, train_num=args.train_num,
            drop_last=args.drop_last, shuffle=args.shuffle, batch_size=args.batch_size)
        self.device = args.device
        self.G = Generator().to(self.device)
        self.Er = Encoder().to(self.device)
        self.Ef = Encoder().to(self.device)
        self.Dx = DiscriminatorX().to(self.device)
        # self.Dy = DiscriminatorY().to(self.device)
        self.Dz = DiscriminatorZ().to(self.device)
        self.Da = DiscriminatorA().to(self.device)
        self.cur_batch_size = args.batch_size
        self.init_models()

    def train(self):
        optimizer_G = optim.Adam(self.G.parameters(), lr=self.args.G_lr, betas=(0.5, 0.999))
        optimizer_Dz = optim.Adam(self.Dz.parameters(), lr=self.args.Dz_lr, betas=(0.5, 0.999))
        optimizer_Ef = optim.Adam(self.Ef.parameters(), lr=self.args.E_lr, betas=(0.5, 0.999))
        optimizer_Er = optim.Adam(self.Er.parameters(), lr=self.args.E_lr, betas=(0.5, 0.999))
        optimizer_Da = optim.Adam(self.Da.parameters(), lr=self.args.Da_lr, betas=(0.5, 0.999))
        optimizer_Dx = optim.Adam(self.Dx.parameters(), lr=self.args.Dx_lr, betas=(0.5, 0.999))

        loss_Da = loss_G = loss_Ef = loss_Er = loss_Dz = loss_Dx = None
        for ep in range(self.args.epoch):
            for data, _, _ in Bar(dataloader=self.dataloader):
                self.cur_batch_size = data.shape[0]
                data = data.to(self.device)
                data = data.reshape(self.cur_batch_size, 1, 73)

                #####
                # 1 #   Discriminator ADV:  max  D(x) - D(G(z))
                #####

                for i in range(self.args.iter_D):
                    noise = torch.randn(self.cur_batch_size, 1, 16).to(self.device)
                    fake = self.G.forward(noise)
                    critic_real = self.Da(data).reshape(-1)
                    critic_fake = self.Da(fake).reshape(-1)
                    gp = gradient_penalty_adv(self.Da, data, fake, self.device)
                    loss_Da = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.args.lambdas * gp
                    )

                    self.Da.zero_grad()
                    loss_Da.backward()
                    optimizer_Da.step()

                #####
                # 2 #    Generator: max D(G(z))
                #####

                for _ in range(1):
                    noise = torch.randn(self.cur_batch_size, 1, 16).to(self.device)
                    fake = self.G(noise)
                    gen_fake = self.Da(fake).reshape(-1)
                    loss_G = -torch.mean(gen_fake)

                    self.G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()

                #####
                # 3 #    Discriminator ZZ: max D(z,z) - D(z,z_rec)
                #####

                for _ in range(self.args.iter_D):
                    noise = torch.randn(self.cur_batch_size, 1, 16, requires_grad=True).to(self.device)
                    fake = self.G(noise)
                    z_rec = self.Ef(fake)
                    critic_real = self.Dz(noise, noise).reshape(-1)
                    critic_fake = self.Dz(noise, z_rec).reshape(-1)
                    gp = gradient_penalty_zz(self.Dz, noise, z_rec, self.device)
                    loss_Dz = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.args.lambdas * gp
                    )

                    self.Dz.zero_grad()
                    loss_Dz.backward()
                    optimizer_Dz.step()

                #####
                # 4 #    Encoder Fake: max D(z,z_rec)
                #####

                for _ in range(self.args.iter_E):
                    noise = torch.randn(self.cur_batch_size, 1, 16).to(self.device)
                    fake = self.G(noise)
                    z_rec = self.Ef(fake)
                    score = self.Dz(noise, z_rec).reshape(-1)
                    loss_Ef = -torch.mean(score)

                    self.Ef.zero_grad()
                    loss_Ef.backward()
                    optimizer_Ef.step()

                #####
                # 5 #    Discriminator XX: max Dx(x,x) - Dx(x,x_rec)
                #####

                for _ in range(1):
                    x_rec = self.G.forward(self.Er.forward(data))
                    data.requires_grad_()
                    critic_real = self.Dx(data, data).reshape(-1)
                    critic_fake = self.Dx(data, x_rec).reshape(-1)
                    gp = gradient_penalty_xx(self.Dx, data, x_rec, self.device)
                    loss_Dx = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.args.lambdas * gp
                    )

                    self.Dx.zero_grad()
                    loss_Dx.backward()
                    optimizer_Dx.step()

                #####
                # 6 #    Encoder Real: max Dx(x,x_rec)
                #####

                for _ in range(1):
                    x_rec = self.G.forward(self.Er.forward(data))
                    critic_fake = self.Dx(data, x_rec).reshape(-1)
                    loss_Er = -torch.mean(critic_fake)

                    self.Er.zero_grad()
                    loss_Er.backward()
                    optimizer_Er.step()

            print(' epoch[{}/{}], Da: {:4f}, G: {:4f}, Dz: {:4f}, Ef: {:4f}, Dx: {:4f}, Er: {:4f}'.format(
                ep + 1, self.args.epoch, loss_Da, loss_G, loss_Dz, loss_Ef, loss_Dx, loss_Er
            ))
            loss = [ep,
                    get_num(loss_Da), get_num(loss_G),
                    get_num(loss_Dz), get_num(loss_Ef),
                    get_num(loss_Dx), get_num(loss_Er)]
            log = pd.DataFrame([loss])
            log.to_csv('D:/code/model/LOG_{}.csv'.format(self.args.save_name), mode='a', header=False, index=False)
        self.save_weights()

    def init_models(self):
        init_weights((self.G, self.Er, self.Ef, self.Dz, self.Da))

    def save_weights(self):
        state_dict_G = self.G.state_dict()
        state_dict_Er = self.Er.state_dict()
        state_dict_Ef = self.Ef.state_dict()
        state_dict_Dx = self.Dx.state_dict()
        # state_dict_Dy = self.Dy.state_dict()
        state_dict_Dz = self.Dz.state_dict()
        state_dict_Da = self.Da.state_dict()

        torch.save({'Generator': state_dict_G,
                    'Encoder_real': state_dict_Er,
                    'Encoder_fake': state_dict_Ef,
                    'Discriminator_X': state_dict_Dx,
                    # 'Discriminator_Y': state_dict_Dy,
                    'Discriminator_Z': state_dict_Dz,
                    'Discriminator_Adv': state_dict_Da},
                   'D:/code/model/MY_{}'.format(self.args.save_name))

    def load_weights(self, name):
        state_dict = torch.load('D:/code/model/MY_{}'.format(name))

        self.G.load_state_dict(state_dict['Generator'])
        self.Er.load_state_dict(state_dict['Encoder_real'])
        self.Ef.load_state_dict(state_dict['Encoder_fake'])
        self.Dx.load_state_dict(state_dict['Discriminator_X'])
        # self.Dy.load_state_dict(state_dict['Discriminator_Y'])
        self.Dz.load_state_dict(state_dict['Discriminator_Z'])
        self.Da.load_state_dict(state_dict['Discriminator_Adv'])

    def test(self, num=1000):
        test_loader_a = ids_dataloader('D:/code/dataset/cicids2018/dataset2.npz', 'Test1a', 1)
        test_loader_b = ids_dataloader('D:/code/dataset/cicids2018/dataset2.npz', 'Test1b', 1)
        score_a = []
        score_b = []
        for (i, (data, _, _)) in enumerate(test_loader_a):
            if i >= num:
                break
            data = data.to(self.device)
            data = data.reshape(1, 1, 73)
            err = abs(self.G(self.Ef(data)) - data)
            score_a.append(abs(err.sum().detach().cpu().numpy()))

        for (i, (data, _, _)) in enumerate(test_loader_b):
            if i >= num:
                break
            data = data.to(self.device)
            data = data.reshape(1, 1, 73)
            err = abs(self.G(self.Ef(data)) - data)
            score_b.append(abs(err.sum().detach().cpu().numpy()))

        show_density((score_a, score_b), 200)

        return score_a, score_b


def init_weights(models):
    for md in models:
        for layer in md.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)


def gradient_penalty_adv(dis, real, fake, device='cuda:0'):
    cur_size = real.shape[0]
    alpha = torch.rand((cur_size, 1, 1)).repeat(1, 1, 73).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    mixed_scores = dis(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp


def gradient_penalty_zz(dis, z, z_rec, device='cuda:0'):
    cur_size = z.shape[0]
    alpha = torch.rand((cur_size, 1, 1)).repeat(1, 1, 16).to(device)
    interpolated_images = z * alpha + z_rec * (1 - alpha)

    mixed_scores = dis(z, interpolated_images)  # (N, 1)

    gradient = torch.autograd.grad(
        inputs=(z, interpolated_images),
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )  # tuple((N,1,16), (N,1,16))

    gradient1 = gradient[0]
    gradient1 = gradient1.view(gradient1.shape[0], -1)
    gradient_norm1 = gradient1.norm(2, dim=1)
    gp1 = torch.mean((gradient_norm1 - 1) ** 2)

    gradient2 = gradient[1]
    gradient2 = gradient2.view(gradient2.shape[0], -1)
    gradient_norm2 = gradient2.norm(2, dim=1)
    gp2 = torch.mean((gradient_norm2 - 1) ** 2)
    return gp1 + gp2


def gradient_penalty_xx(dis, x, x_rec, device='cuda:0'):
    cur_size = x.shape[0]
    alpha = torch.rand((cur_size, 1, 1)).repeat(1, 1, 73).to(device)
    interpolated_images = x * alpha + x_rec * (1 - alpha)

    mixed_scores = dis(x, interpolated_images)

    gradient = torch.autograd.grad(
        inputs=(x, interpolated_images),
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )
    gradient1 = gradient[0]
    gradient1 = gradient1.view(gradient1.shape[0], -1)
    gradient_norm1 = gradient1.norm(2, dim=1)
    gp1 = torch.mean((gradient_norm1 - 1) ** 2)

    gradient2 = gradient[1]
    gradient2 = gradient2.view(gradient2.shape[0], -1)
    gradient_norm2 = gradient2.norm(2, dim=1)
    gp2 = torch.mean((gradient_norm2 - 1) ** 2)
    return gp1 + gp2


def get_num(loss):
    return loss.detach().cpu().numpy().min()
