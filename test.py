from train import Trainer
from arg import Args
from utils.seed import set_seed
from dataset import ids_dataloader
from utils.plot import show_flow, show_density
from scipy.stats import normaltest, shapiro


def tttt(self, num=1000):
    test_loader_a = ids_dataloader('D:/code/dataset/cicids2018/dataset2.npz', 'Test1a', 1)
    test_loader_b = ids_dataloader('D:/code/dataset/cicids2018/dataset2.npz', 'Test1b', 1)
    score_a = []
    score_b = []
    for (i, (data, _, _)) in enumerate(test_loader_a):
        if i >= num:
            break
        data = data.to(self.device)
        data = data.reshape(1, 1, 73)
        err = self.G(self.Ef(data)) - data
        score_a.append(err.detach().cpu().numpy().reshape(-1))

    for (i, (data, _, _)) in enumerate(test_loader_b):
        if i >= num:
            break
        data = data.to(self.device)
        data = data.reshape(1, 1, 73)
        err = self.G(self.Ef(data)) - data
        score_b.append(err.detach().cpu().numpy().reshape(-1))

    return score_a, score_b


def quantile(data, alpha):
    err = []
    for d in data:
        err.append(sum(abs(d)))
    err.sort(reverse=True)
    num = len(err) * alpha
    if int(num) == num:
        return err[int(num) - 1]
    else:
        return (err[int(len(err) * alpha) - 1] + err[int(len(err) * alpha)]) / 2


if __name__ == '__main__':
    set_seed(1234)
    args = Args(batch_size=25,
                g_lr=1e-4,
                e_lr=1e-4,
                da_lr=1e-4,
                dx_lr=1e-4,
                dz_lr=1e-4,
                epoch=15,
                iter_D=4,
                iter_E=4,
                latent_dimension=16,
                lambdas=10,
                device='cuda:0',
                dataset_dir='D:/code/dataset/cicids2018/dataset2.npz',
                dataset_name='Train',
                train_num=50000,
                save_name='20220822'
                )

    tr = Trainer(args)
    # tr.train()
    tr.load_weights('20220527test1')
    tr.test()

    a, b = tttt(tr)

    stat, p = normaltest(b)
    print(p)
    print(stat)
    stat, p = shapiro(b)
    print(p)
    print(stat)

    sc = quantile(b, 0.02)
    c = sum(1 for i in a if sum(abs(i)) < sc)
    acc = 1 - (c + 20) / 2000
    print(acc)


