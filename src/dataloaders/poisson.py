import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class PoissonDataset(Dataset):
    def __init__(self, data_path):
        
        data = loadmat(data_path)
        phi = data["phi_data"].astype("float32")  # (N, 64, 64)
        f   = data["f_data"].astype("float32")

        # Always normalize the data
        u_mean, u_std = phi.mean(), phi.std() + 1e-8
        f_mean, f_std = f.mean(),   f.std() + 1e-8
        phi = (phi - u_mean) / u_std
        f   = (f   - f_mean) / f_std
        
        self.u = torch.from_numpy(phi).contiguous()
        self.f = torch.from_numpy(f).contiguous()
        
    def __len__(self): return self.u.shape[0]

    def __getitem__(self, index): return self.u[index], self.f[index]