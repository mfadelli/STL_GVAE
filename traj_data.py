import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    #is the matrix of trajectories, y is the array of labels
    def __init__(self, device, x=None, y=None):
        self.trajectories = x.to(device)
        self.labels = y.to(device)
        self.nvars = x.shape[1]
        self.npoints = x.shape[-1]
        self.mean = torch.zeros(self.nvars).to(device)
        self.std = torch.zeros(self.nvars).to(device)
        self.normalized = False

    def reshape_mean_std(self):
        rep_mean = torch.cat([self.mean[i].repeat(
            self.trajectories.shape[0], self.trajectories.shape[-1]).unsqueeze(1) for i in range(self.nvars)], dim=1)
        rep_std = torch.cat([self.std[i].repeat(
            self.trajectories.shape[0], self.trajectories.shape[-1]).unsqueeze(1) for i in range(self.nvars)], dim=1)
        return rep_mean.to(self.trajectories.device), rep_std.to(self.trajectories.device)

    def normalize(self):
        self.mean = torch.tensor([self.trajectories[:, i, :].mean() for i in range(self.nvars)])
        self.std = torch.tensor([self.trajectories[:, i, :].std() for i in range(self.nvars)])
        rep_mean, rep_std = self.reshape_mean_std()
        self.trajectories = (self.trajectories - rep_mean) / rep_std
        self.normalized = True

    def inverse_normalize(self):
        rep_mean, rep_std = self.reshape_mean_std()
        self.trajectories = (self.trajectories * rep_std) + rep_mean
        self.normalized = False

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]