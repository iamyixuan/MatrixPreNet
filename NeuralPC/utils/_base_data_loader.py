from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataLoader(ABC):
    def __init__(self, data_dir, batch_size, shuffle, validation_split):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.init()

    @abstractmethod
    def init(self):
        pass

    def split_validation(self):
        if self.validation_split == 0.0:
            return None
        else:
            return self.train_loader, self.valid_loader

class BaseDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]
