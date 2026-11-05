import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.utils.convert import from_networkx
import os
from tqdm import tqdm

torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        name="no_param",
        root="data/BREC",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        self.root = root
        self.name = name
        print("Force Reload is: ", force_reload)
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3.pt"]

    def process(self):
        print("Processing BREC Dataset is called...")
        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = [graph6_to_pyg(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list, desc="Pre-transforming")]

        for data in data_list:
            data.x = torch.ones(data.num_nodes, 1)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BRECDiskDataset(Dataset):
    """Disk-backed variant of the BREC dataset.

    Loads raw graph6 bytes in memory once, then processes each graph one-by-one,
    applying pre_filter / pre_transform and saving each as its own file to avoid
    holding the entire processed dataset in RAM.
    """

    def __init__(
        self,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        progress: bool = True,
    ):
        self.root = root
        self.name = name
        self._progress = progress
        super().__init__(root, transform, pre_transform, pre_filter, log=False, force_reload=force_reload)
        
        # Load meta to know length
        meta = torch.load(self.processed_paths[0], weights_only=False)
        self._len = int(meta["length"])

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_dir(self):
        name = "processed_disk"
        return os.path.join(self.root, self.name, name)

    @property
    def processed_file_names(self):
        # single meta file to mark completion + store length
        return ["meta.pt"]

    def len(self) -> int:
        return self._len

    def get(self, idx: int):
        path = os.path.join(self.processed_dir, f"data_{idx}.pt")
        data = torch.load(path, weights_only=False)
        return data

    def process(self):
        raw = np.load(self.raw_paths[0], allow_pickle=True)

        save_idx = 0
        iterator = enumerate(raw)
        if self._progress:
            iterator = tqdm(iterator, total=len(raw), desc="Processing BREC (disk)")

        for i, g6 in iterator:
            data = graph6_to_pyg(g6)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # ensure node features
            if not hasattr(data, "x") or data.x is None:
                data.x = torch.ones(data.num_nodes, 1)

            torch.save(data, os.path.join(self.processed_dir, f"data_{save_idx}.pt"))
            save_idx += 1

        torch.save({"length": save_idx}, self.processed_paths[0])


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
