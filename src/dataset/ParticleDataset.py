from torch.utils.data import Dataset
import h5py
import os
import numpy as np


class ParticleDataset(Dataset):
    def __init__(self, directory_path, num_jets=None):
        # Initialize data containers
        features_list = []
        labels_list = []

        # Loop through files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".hdf5") or filename.endswith(".h5"):
                file_path = os.path.join(directory_path, filename)
                with h5py.File(file_path, "r") as hdf:
                    print(f"Loading {directory_path}/{filename}")
                    features = hdf["features"][:]
                    labels = hdf["labels"][:]

                labels_list.append(labels)
                features_list.append(features)

                print(f"Loaded {directory_path}/{filename}")

        # Concatenate all the data
        self.features = np.concatenate(features_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)

        if num_jets:
            self.labels = self.labels[:num_jets]
            self.features = self.features[:num_jets]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
