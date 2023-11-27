from train import Trainer
from arg import Args


if __name__ == '__main__':
    args = Args(batch_size=25,
                g_lr=1e-4,
                e_lr=1e-4,
                da_lr=1e-4,
                dz_lr=1e-4,
                epoch=100,
                iter_D=4,
                iter_E=4,
                latent_dimension=16,
                lambdas=10,
                device='cuda:0',
                dataset_dir='D:/code/dataset/cicids2018/dataset2.npz',
                dataset_name='Train',
                train_num=50000,
                save_name='20220520_1'
                )

    trainer = Trainer(args)
    trainer.train()
    # save
    # test
    print('Training is completed')
