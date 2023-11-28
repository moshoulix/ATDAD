import logging
from nets import Generator, Encoder, DiscriminatorA, DiscriminatorZ, DiscriminatorX
from dataset import ids_dataloader
# from utils import show_density, show_flow
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim


class Trainer:
    def __init__(self, args):
        self.args = args
        self.dataloader = ids_dataloader(
            npz='./dataset/handled_dataset2.npz', name='Train', train_num=args.train_num,
            drop_last=args.drop_last, shuffle=args.shuffle, batch_size=args.batch_size)
        self.device = args.device
        self.G = Generator().to(self.device)
        self.Er = Encoder().to(self.device)
        self.Ef = Encoder().to(self.device)
        self.Dx = DiscriminatorX().to(self.device)
        self.Dz = DiscriminatorZ().to(self.device)
        self.Da = DiscriminatorA().to(self.device)
        self.cur_batch_size = args.batch_size
        self.init_models()
        if args.load is not None:
            self.load_weights(args.load)

    def train(self):
        if not os.path.exists('./log'):
            os.makedirs('./log')

        now = datetime.now()
        logging.basicConfig(filename='./log/{}_{}_training.log'.format(self.args.save_name, now.strftime('%Y%m%d_%H_%M')),
                            level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        optimizer_Dz = optim.Adam(self.Dz.parameters(), lr=self.args.Dz_lr, betas=(0.5, 0.999))
        optimizer_Er = optim.Adam(self.Er.parameters(), lr=self.args.E_lr, betas=(0.5, 0.999))
        optimizer_Da = optim.Adam(self.Da.parameters(), lr=self.args.Da_lr, betas=(0.5, 0.999))
        optimizer_Dx = optim.Adam(self.Dx.parameters(), lr=self.args.Dx_lr, betas=(0.5, 0.999))
        optimizer_Ef_G = optim.Adam(list(self.Ef.parameters()) + list(self.G.parameters()), lr=self.args.E_lr, betas=(0.5, 0.999))

        loss_Da = loss_G = loss_Ef = loss_Er = loss_Dz = loss_Dx = None
        print('Trainning begins.')

        for ep in range(self.args.epoch):
            for data, _, _ in self.dataloader:
                self.cur_batch_size = data.shape[0]
                data = data.to(self.device)
                data = data.reshape(self.cur_batch_size, 1, 73)
                self.set_train_mode(True)
                #####
                # 1 #   Discriminator ADV:  max  Da(x) - Da(G(z))
                #####

                for i in range(self.args.iter_D):
                    noise = torch.randn(self.cur_batch_size, 1, self.args.latent_dim).to(self.device)
                    fake = self.G.forward(noise)
                    critic_real = self.Da(data).reshape(-1)
                    critic_fake = self.Da(fake).reshape(-1)
                    gp = gradient_penalty_adv(self.Da, data, fake, self.args.data_dim, self.device)
                    loss_Da = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.args.lambdas * gp
                    )

                    self.Da.zero_grad()
                    loss_Da.backward()
                    optimizer_Da.step()

                #####
                # 2 #    Discriminator ZZ: max D(z,z) - D(z,z_rec)
                #####

                for _ in range(self.args.iter_D):
                    noise = torch.randn(self.cur_batch_size, 1, self.args.latent_dim, requires_grad=True).to(
                        self.device)
                    fake = self.G(noise)
                    z_rec = self.Ef(fake)
                    critic_real = self.Dz(noise, noise).reshape(-1)
                    critic_fake = self.Dz(noise, z_rec).reshape(-1)
                    gp = gradient_penalty_zz(self.Dz, noise, z_rec, self.args.latent_dim, self.device)
                    loss_Dz = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.args.lambdas * gp
                    )

                    self.Dz.zero_grad()
                    loss_Dz.backward()
                    optimizer_Dz.step()

                #####
                # 3 #    Discriminator XX: max Dx(x,x) - Dx(x,x_rec)
                #####

                for _ in range(1):
                    x_rec = self.G.forward(self.Er.forward(data))
                    data.requires_grad_()
                    critic_real = self.Dx(data, data).reshape(-1)
                    critic_fake = self.Dx(data, x_rec).reshape(-1)
                    gp = gradient_penalty_xx(self.Dx, data, x_rec, self.args.data_dim, self.device)
                    loss_Dx = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.args.lambdas * gp
                    )

                    self.Dx.zero_grad()
                    loss_Dx.backward()
                    optimizer_Dx.step()

                #####
                # 4 #    Encoder Fake: max D(z,z_rec)  |   Generator: max D(G(z))
                #####

                for _ in range(self.args.iter_E):
                    noise = torch.randn(self.cur_batch_size, 1, self.args.latent_dim).to(self.device)
                    fake = self.G(noise)
                    gen_fake = self.Da(fake).reshape(-1)
                    z_rec = self.Ef(fake)
                    score = self.Dz(noise, z_rec).reshape(-1)
                    loss_Ef = -torch.mean(score)
                    loss_G = -torch.mean(gen_fake)
                    loss_Ef_G = loss_Ef + loss_G

                    optimizer_Ef_G.zero_grad()
                    loss_Ef_G.backward()
                    optimizer_Ef_G.step()

                #####
                # 5 #    Encoder Real: max Dx(x,x_rec)
                #####

                for _ in range(self.args.iter_E):
                    x_rec = self.G.forward(self.Er.forward(data))
                    critic_fake = self.Dx(data, x_rec).reshape(-1)
                    loss_Er = -torch.mean(critic_fake)

                    self.Er.zero_grad()
                    loss_Er.backward()
                    optimizer_Er.step()

            print(' epoch[{}/{}], Da: {:4f}, G: {:4f}, Dz: {:4f}, Ef: {:4f}, Dx: {:4f}, Er: {:4f}'.format(
                ep + 1, self.args.epoch, loss_Da, loss_G, loss_Dz, loss_Ef, loss_Dx, loss_Er
            ))
            logging.info(' epoch[{}/{}], Da: {:4f}, G: {:4f}, Dz: {:4f}, Ef: {:4f}, Dx: {:4f}, Er: {:4f}'.format(
                ep + 1, self.args.epoch, loss_Da, loss_G, loss_Dz, loss_Ef, loss_Dx, loss_Er
            ))
            if (ep + 1) % 10 == 0 and (ep + 1) != self.args.epoch:
                res = self.test()
                logging.info('acc: {}, pre: {}, F1: {}, recall: {}'
                             .format(res['acc'], res['pre'], res['F1'], res['recall']))
                # self.save()
        logging.info('Finished.')

    def init_models(self):
        init_weights((self.G, self.Er, self.Ef, self.Dz, self.Da))

    def save(self):
        if not os.path.exists('./saved_models'):
            os.makedirs('./saved_models')

        state_dict_G = self.G.state_dict()
        state_dict_Er = self.Er.state_dict()
        state_dict_Ef = self.Ef.state_dict()
        state_dict_Dx = self.Dx.state_dict()
        state_dict_Dz = self.Dz.state_dict()
        state_dict_Da = self.Da.state_dict()
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        model_filename = f'{self.args.save_name}_{current_time}.pth'

        torch.save({'Generator': state_dict_G,
                    'Encoder_real': state_dict_Er,
                    'Encoder_fake': state_dict_Ef,
                    'Discriminator_X': state_dict_Dx,
                    'Discriminator_Z': state_dict_Dz,
                    'Discriminator_Adv': state_dict_Da},
                   './saved_models/{}'.format(model_filename))
        print('Saved as {}'.format(model_filename))
        return model_filename

    def load_weights(self, name):
        state_dict = torch.load('./saved_models/{}'.format(name))

        self.G.load_state_dict(state_dict['Generator'])
        self.Er.load_state_dict(state_dict['Encoder_real'])
        self.Ef.load_state_dict(state_dict['Encoder_fake'])
        self.Dx.load_state_dict(state_dict['Discriminator_X'])
        self.Dz.load_state_dict(state_dict['Discriminator_Z'])
        self.Da.load_state_dict(state_dict['Discriminator_Adv'])

    def test(self, num_a=1000, num_b=1000, s=0.02):
        """"""
        self.set_train_mode(False)
        test_loader_a = ids_dataloader('./dataset/handled_dataset2.npz', 'Test1a', 1)
        test_loader_b = ids_dataloader('./dataset/handled_dataset2.npz', 'Test1b', 1)
        score_a = []
        score_b = []

        with torch.no_grad():
            for (i, (data, _, _)) in enumerate(test_loader_a):
                if i >= num_a:
                    break
                data = data.to(self.device)
                data = data.reshape(1, 1, self.args.data_dim)
                err = abs(self.G(self.Er(data)) - data)
                score_a.append(abs(err.sum().detach().cpu().numpy()))

            for (i, (data, _, _)) in enumerate(test_loader_b):
                if i >= num_b:
                    break
                data = data.to(self.device)
                data = data.reshape(1, 1, self.args.data_dim)
                err = abs(self.G(self.Er(data)) - data)
                score_b.append(abs(err.sum().detach().cpu().numpy()))

        # show_density((score_a, score_b), 200)
        score_b.sort()
        cls = score_b[int(num_b * (1-s)) - 1]
        temp = sum(1 for i in score_a if i < cls) / num_a
        recall = sum(1 for i in score_b if i < cls) / num_b
        pre = recall * num_b / (recall * num_b + temp * num_a)
        f = 2 * pre * recall / (recall + pre)
        acc = (1 - temp + recall) / 2
        print('acc:{:.4f}, F1:{:.4f}, recall:{:.4f}, pre:{:.4f}'.format(acc, f, recall, pre))
        return {'score_a': score_a, 'score_b': score_b,
                'acc': acc, 'pre': pre, 'F1': f, 'recall': recall, 'cls': cls, 'FP': temp}

    def set_train_mode(self, mode):
        if mode:
            self.G.train()
            self.Ef.train()
            self.Er.train()
            self.Da.train()
            self.Dz.train()
            self.Dx.train()
        else:
            self.G.eval()
            self.Ef.eval()
            self.Er.eval()
            self.Da.eval()
            self.Dz.eval()
            self.Dx.eval()


def init_weights(models):
    for md in models:
        for layer in md.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)


def gradient_penalty_adv(dis, real, fake, data_dim, device='cuda:0'):
    cur_size = real.shape[0]
    alpha = torch.rand((cur_size, 1, 1)).repeat(1, 1, data_dim).to(device)
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


def gradient_penalty_zz(dis, z, z_rec, latent_dim, device='cuda:0'):
    cur_size = z.shape[0]
    alpha = torch.rand((cur_size, 1, 1)).repeat(1, 1, latent_dim).to(device)
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


def gradient_penalty_xx(dis, x, x_rec, data_dim, device='cuda:0'):
    cur_size = x.shape[0]
    alpha = torch.rand((cur_size, 1, 1)).repeat(1, 1, data_dim).to(device)
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
