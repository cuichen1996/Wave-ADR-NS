import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data.data import KappaDataGenerator

class MinMaxScalerVectorized:
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, tensor):
        """Fit features

        Parameters
        ----------
        stacked_features : tuple, list
            List of stacked features.

        Returns
        -------
        tensor
            A tensor with scaled features using requested preprocessor.
        """
        # Feature range
        a, b = self.feature_range
        return (tensor - tensor.min()) * (b - a) / (tensor.max() - tensor.min()) + a

   

class CreatDataset(torch.utils.data.Dataset):
    def __init__(self, p, data1, data2):
        self.p = p
        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, index):
        r = self.data1[index, :]
        e = self.data2[index, :]
        return r, e

    def __len__(self):
        return self.p


def CreateDataLoader(config):
    N = 128
    kappa = torch.from_numpy(np.load("data/kappa128.npy"))
    data_num = len(kappa)
    print("Number of training data: ", data_num)
    f = torch.zeros_like(kappa, dtype=torch.cfloat)
    f[:,:, N//2,N//2] = 1.0 * N**2
    trainset = CreatDataset(data_num, f, kappa)
    train_loader128 = torch.utils.data.DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(trainset),
    )

    N = 256
    kappa = torch.from_numpy(np.load("/data/kappa256.npy"))
    data_num = len(kappa)
    print("Number of training data: ", data_num)
    f = torch.zeros_like(kappa, dtype=torch.cfloat)
    f[:,:, N//2,N//2] = 1.0 * N**2
    trainset = CreatDataset(data_num, f, kappa)
    train_loader256 = torch.utils.data.DataLoader(
        trainset,
        batch_size=config["batch_size"]//2,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(trainset),
    )

    N = 512
    kappa = torch.from_numpy(np.load("data/kappa512.npy"))
    data_num = len(kappa)
    print("Number of training data: ", data_num)
    f = torch.zeros_like(kappa, dtype=torch.cfloat)
    f[:,:, N//2,N//2] = 1.0 * N**2
    trainset = CreatDataset(data_num, f, kappa)
    train_loader512 = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size']//4, shuffle=False, pin_memory=True, sampler=DistributedSampler(trainset))

    return train_loader128, train_loader256, train_loader512


def CreateTestLoader(N, batch, DATASET, src):
    if DATASET == "cifar10":
        dataset = KappaDataGenerator(N, N)
        dataset.load_data(data="cifar10")

        kappa = torch.zeros((batch, 1, N, N))
        for i in range(batch):
            kappa[i, 0] = dataset.generate_kappa().reshape(N, N)
    elif DATASET == "stl10":
        dataset = KappaDataGenerator(N, N)
        dataset.load_data(data="stl10")

        kappa = torch.zeros((batch, 1, N, N))
        for i in range(batch):
            kappa[i, 0] = dataset.generate_kappa().reshape(N, N)
    elif DATASET == "styleA" or DATASET == "styleB":
        velocity = np.load('data/'+DATASET+'.npy')
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((N, N)),
            transforms.ToTensor(),
            MinMaxScalerVectorized(feature_range=(0.25,1)),
        ])
        kappa = []
        for i in range(batch):
            sample=np.random.randint(0,velocity.shape[0])
            img_tensor = transform(velocity[sample,0].astype(np.float32))[0]
            kappa.append(img_tensor.unsqueeze(0).unsqueeze(0))
        kappa = torch.cat(kappa, dim=0)

    b = torch.zeros_like(kappa, dtype=torch.cfloat)
    b[:,:, src[0], src[1]] = 1.0 * N**2
 
    testset = CreatDataset(batch, b, kappa)
    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False
    )
    return test_loader