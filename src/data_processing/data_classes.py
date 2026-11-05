import os
import torch
import psutil

n_workers = min(58, (psutil.cpu_count(logical=False) - 1))
print(f"Using {n_workers} workers for data processing.")


from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch
from torch_geometric.data import Dataset, Data
import os.path as osp
import random
import tqdm
from torch_geometric.datasets import TUDataset, MalNetTiny

import io
import torch
import numpy as np


#### SOME DEBUGGING HELPERS ####
def serialized_size_mb(obj) -> float:
    """Return size in MB of how large this object would be when torch.save'd."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    size_bytes = buffer.tell()
    return size_bytes / (1024 ** 2)


def inspect_data_tensors(data):
    """
    Print tensor attributes and their approximate in-memory size in MB.
    Works well for torch_geometric.data.Data objects.
    """
    for k, v in data.__dict__.items():
        if torch.is_tensor(v):
            size_mb = v.numel() * v.element_size() / (1024 ** 2)
            print(f"{k:20s} {tuple(v.shape)!s:20s} {size_mb:8.2f} MB")
        else:
            print(f"{k:20s} {type(v)}")


def detach_from_storage(data):
    # Recreate the same data class with cloned tensors so custom properties
    # (e.g. SubgraphData.num_subgraphs) remain available after detaching from
    # shared-memory-backed storages.
    out = data.__class__()
    for key, value in data:
        if torch.is_tensor(value):
            setattr(out, key, value.clone())  # new, compact storage
        else:
            setattr(out, key, value)
    return out



### multiprocessing helper

def _process_one(raw_path, pre_filter, pre_transform):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    try:
        data = torch.load(raw_path, weights_only=False)
    except Exception as e:
        return None, f"Error loading {raw_path}: {e}"

    try:
        if pre_filter is not None and not pre_filter(data):
            return None, None
    except Exception as e:
        return None, f"Error in pre_filter for {raw_path}: {e}"

    try:
        if pre_transform is not None:
            data = pre_transform(data)
    except Exception as e:
        return None, f"Error in pre_transform for {raw_path}: {e}"

    return data, None

def _process_one_tudataset(data, pre_filter, pre_transform):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    if pre_filter is not None and not pre_filter(data):
        return None

    if pre_transform is not None:
        data = pre_transform(data)

    return data


def _adaptive_chunksize(total_tasks: int, workers: int, max_chunksize: int = 16) -> int:
    """
    Small adaptive chunksize to reduce scheduling overhead without hurting load balancing.
    """
    if total_tasks <= 0 or workers <= 0:
        return 1
    if total_tasks <= workers * 4:
        return 1
    # Aim for multiple chunks per worker to keep balancing responsive.
    return max(1, min(max_chunksize, total_tasks // (workers * 8)))


def _process_malnet_file(task):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    path, y = task
    try:
        with open(path) as f:
            edges = f.read().split('\n')[5:-1]
        edge_indices = [[int(s) for s in e.split()] for e in edges]
        edge_index = torch.tensor(edge_indices).t().contiguous()
        num_nodes = int(edge_index.max()) + 1
        data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
        return data, None
    except Exception as e:
        return None, f"Error processing {path}: {e}"

def _process_malnet_file_with_transforms(task, pre_filter, pre_transform):
    data, err = _process_malnet_file(task)
    if err is not None or data is None:
        return data, err
    try:
        if pre_filter is not None and not pre_filter(data):
            return None, None
    except Exception as e:
        return None, f"Error in pre_filter: {e}"

    try:
        if pre_transform is not None:
            data = pre_transform(data)
    except Exception as e:
        return None, f"Error in pre_transform: {e}"

    return data, None


class TUDiskDataset(Dataset):
    """ Torch Geometric Dataset class for loading TUDatasets but processing them on disk one by one."""
    def __init__(self, name, root, transform=None, pre_transform=None, pre_filter=None, multiprocess=False, force_reload=True):
        self.name = name
        self.multiprocess = multiprocess
        self.idx = len(os.listdir(root + '/processed')) - 2 if os.path.exists(root + '/processed') else 0
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)


    @property
    def raw_file_names(self):
        return TUDataset(self.root, name=self.name).raw_file_names

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.idx)]
    
    def process(self):
        if self.multiprocess:
            self.multiprocess_process()
        else:
            self.sequential_process()

    def sequential_process(self):
        self.idx = 0
        print(f"Processing raw data with \n PRE-FILTERS: {self.pre_filter} \n PRE-TRANSFORMS; {self.pre_transform}")

        dataset = TUDataset(self.root, name=self.name)

        for i in tqdm.tqdm(range(len(dataset)), desc=f"Processing graphs on disk"):
            data = detach_from_storage(dataset[i])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            # transform and save processed data object
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{self.idx}.pt'))
            self.idx += 1
            print("Single-Processed {} graphs so far.".format(self.idx))

        print(f"Loaded {self.idx} graphs.")

    def multiprocess_process(self):
        self.idx = 0
        print(f"Multi-Processing with {n_workers} workers for raw data with \n PRE-FILTERS: {self.pre_filter} \n PRE-TRANSFORMS; {self.pre_transform}")
        
        dataset = TUDataset(self.root, name=self.name)
        data_list = [detach_from_storage(dataset[i]) for i in range(len(dataset))]
        print("Prepared the datalist for multiprocessing.")

        worker = partial(
                _process_one_tudataset,
                pre_filter=self.pre_filter,
                pre_transform=self.pre_transform,
            )

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            chunksize = _adaptive_chunksize(len(data_list), n_workers)
            for data in tqdm.tqdm(
                ex.map(worker, data_list, chunksize=chunksize),
                total=len(data_list),
                desc=f"Multiprocessing graphs on disk with {n_workers} workers."
            ):
                if data is None:
                    # graph was filtered out
                    continue

                torch.save(data, osp.join(self.processed_dir, f"data_{self.idx}.pt"))
                self.idx += 1

                if self.idx % 1_000 == 0:
                    print(f"Multi-Processed {self.idx} graphs so far.")

        print(f"Loaded {self.idx} graphs.")

    def len(self):
        return self.idx

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)

        if self.transform:
            data = self.transform(data)

        return data


class GraphNet(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, subsample_graphs: int=None, pre_filter=None, 
                 multiprocess=False, force_reload=True):
        self.multiprocess = multiprocess
        self.subsample_graphs = subsample_graphs
        self.generator = random.Random(42)
        self.idx = len(os.listdir(root + '/processed')) - 2  if os.path.exists(root + '/processed') else 0
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.idx)]
    
    def process(self):
        if self.multiprocess:
            self.multiprocess_process()
        else:
            self.sequential_process()

    def sequential_process(self):
        self.idx = 0
        print(f"Processing raw data with {n_workers} workers for \n PRE-FILTERS: {self.pre_filter} \n PRE-TRANSFORMS; {self.pre_transform}")

        if self.subsample_graphs is not None:
            selected_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
            self.generator.shuffle(selected_files)
        else:
            selected_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

        num_files = str(self.subsample_graphs) if self.subsample_graphs is not None else "all"
        for raw_path in tqdm.tqdm(selected_files, desc=f"Scanning to select {num_files} graphs"):

            try:
                data = torch.load(raw_path, weights_only=False)
            except Exception as e:
                print(f"Error loading {raw_path}: {e}")
                continue

            try:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
            except Exception as e:
                print(f"Error in pre_filter for {raw_path}: {e}")
                continue
            
            # transform and save processed data object
            try:
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
            except Exception as e:
                print(f"Error in pre_transform for {raw_path}: {e}")
                continue

            torch.save(data, osp.join(self.processed_dir, f'data_{self.idx}.pt'))
            self.idx += 1

            if self.idx % 1_000 == 0:
                print(f"Processed {self.idx} graphs so far.")

            if self.subsample_graphs is not None and self.idx >= self.subsample_graphs:
                break

        print(f"Loaded {self.idx} graphs.")


    def multiprocess_process(self):
        self.idx = 0
        print(f"Multi-Processing raw data with {n_workers} workers for \n PRE-FILTERS: {self.pre_filter} \n PRE-TRANSFORMS; {self.pre_transform}")

        if self.subsample_graphs is not None:
            selected_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
            self.generator.shuffle(selected_files)
        else:
            selected_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

        num_files = str(self.subsample_graphs) if self.subsample_graphs is not None else "all"

        worker = partial(
            _process_one,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
        )

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            chunksize = _adaptive_chunksize(len(selected_files), n_workers)
            for data, err in tqdm.tqdm(
                ex.map(worker, selected_files, chunksize=chunksize),
                total=len(selected_files),
                desc=f"Multi-Scanning with {n_workers} workers to select {num_files} graphs"
            ):
                if err is not None:
                    print(err)
                    continue
                if data is None:
                    continue

                torch.save(data, osp.join(self.processed_dir, f"data_{self.idx}.pt"))
                self.idx += 1

                if self.idx % 1_000 == 0:
                    print(f"Processed {self.idx} graphs so far.")

                if self.subsample_graphs is not None and self.idx >= self.subsample_graphs:
                    break

    def len(self):
        return self.idx

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)

        if self.transform:
            data = self.transform(data)

        return data
    

class MalNetTinyCustom(MalNetTiny):
    def __init__(
        self,
        root,
        split=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        multiprocess=False,
        num_workers=None,
    ):
        self.multiprocess = multiprocess
        self.num_workers = n_workers if num_workers is None else num_workers
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

    def process(self):
        y_map = {}
        data_list = []
        split_slices = [0]

        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_paths[1], f'{split}.txt')) as f:
                filenames = f.read().split('\n')[:-1]
                split_slices.append(split_slices[-1] + len(filenames))

            for filename in filenames:
                malware_type = filename.split('/')[0]
                if malware_type not in y_map:
                    y_map[malware_type] = len(y_map)

            if self.multiprocess:
                tasks = []
                for filename in filenames:
                    path = osp.join(self.raw_paths[0], f'{filename}.edgelist')
                    y = y_map[filename.split('/')[0]]
                    tasks.append((path, y))

                worker = partial(
                    _process_malnet_file_with_transforms,
                    pre_filter=self.pre_filter,
                    pre_transform=self.pre_transform,
                )
                with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                    chunksize = _adaptive_chunksize(len(tasks), self.num_workers)
                    for data, err in tqdm.tqdm(
                        ex.map(worker, tasks, chunksize=chunksize),
                        total=len(tasks),
                        desc=f"Processing {split} split",
                    ):
                        if err is not None:
                            print(err)
                            continue
                        if data is None:
                            continue
                        data_list.append(data)
            else:
                for filename in tqdm.tqdm(
                    filenames,
                    desc=f"Processing {split} split",
                ):
                    path = osp.join(self.raw_paths[0], f'{filename}.edgelist')
                    y = y_map[filename.split('/')[0]]
                    data, err = _process_malnet_file((path, y))
                    if err is not None:
                        print(err)
                        continue
                    if data is None:
                        continue
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)

        self.save(data_list, self.processed_paths[0])
        torch.save(split_slices, self.processed_paths[1])
