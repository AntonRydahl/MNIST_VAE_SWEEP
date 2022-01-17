import torch
from torch.utils.data import DataLoader


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, file_name, path='data/processed/', verbose=False):
        data = torch.load(path+file_name)
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return torch.tensor(self.images[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
        return self.images[idx], self.labels[idx]


def TrainLoader(batch_size):
    train_dataset = CustomDataSet('train.pt')
    return DataLoader(dataset=train_dataset,
                            batch_size=batch_size, shuffle=True)

def TestLoader(batch_size):
    test_dataset = CustomDataSet('test.pt')
    return DataLoader(dataset=test_dataset,
                            batch_size=batch_size, shuffle=False)