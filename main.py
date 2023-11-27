from train import Trainer
import argparse
from utils import set_seed


parser = argparse.ArgumentParser(description='PyTorch CIC-IDS-2018 Training')

parser.add_argument('--device', default='cuda:0', type=str, metavar='N', help='gpu is all you need')
parser.add_argument('--batch_size', default=25, type=int, metavar='N', help='batchsize (default: 25)')
parser.add_argument('--g_lr', default=1e-4, type=float, metavar='N', help='learning rate of generator')
parser.add_argument('--e_lr', default=1e-4, type=float, metavar='N', help='learning rate of encoder')
parser.add_argument('--da_lr', default=1e-4, type=float, metavar='N', help='learning rate of discriminator A')
parser.add_argument('--dz_lr', default=1e-4, type=float, metavar='N', help='learning rate of discriminator Z')
parser.add_argument('--epoch', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--iter_D', default=4, type=int, metavar='N',
                    help='number of times the generator is trained per one training of the discriminator')
parser.add_argument('--iter_E', default=4, type=int, metavar='N',
                    help='number of times the generator is trained per one training of the encoder')
parser.add_argument('--latent_dimension', default=16, type=int, metavar='N',
                    help='dimension of the latent layer')
parser.add_argument('--lambdas', default=10, type=float, metavar='N',
                    help='parameters of gradient penalty')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--dataset_dir', default='./dataset/handled_dataset2.npz', type=str)
parser.add_argument('--dataset_name', default='Train', type=str)
parser.add_argument('--train_num', default=50000, type=int)
parser.add_argument('--load', default=None, type=str, help='the name of the checkpoint in ./saved_models, if needed')
parser.add_argument('--saved_name', default=None, type=str, help='the name of model')

args = parser.parse_args()
if __name__ == '__main__':
    if args.seed is not None:
        set_seed(args.seed)
    trainer = Trainer(args)
    trainer.train()
    trainer.save()
    trainer.test()
    print('Completed')
