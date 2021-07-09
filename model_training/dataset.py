from torch.utils.data import Dataset
import pickle


class FinCLData(Dataset):
    """"""

    def __init__(self, data_path):
        """
        data_path: path to the data pickle file.
        """
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        temp = self.data[idx]
        return temp
