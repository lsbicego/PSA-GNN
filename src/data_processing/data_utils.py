from torch_geometric.data import Data
from torch_geometric.transforms import Compose, ComposeFilters, BaseTransform
import torch_geometric.transforms as T
import torch
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
from omegaconf import OmegaConf

def compute_subgraph_statistics(dataset):
    """Aggregate dataset-level statistics over precomputed subgraphs.

    Args:
        dataset: Iterable of ``SubgraphData`` objects produced by the subgraph
            preprocessing pipeline.

    Returns:
        A dictionary with average counts and duplication ratios for subgraphs,
        nodes, overlaps, and edges.
    """

    total_graphs = 0

    total_subgraphs = 0
    total_subgraph_nodes = 0.0
    total_subgraph_size_frac = 0.0

    total_overlap_nodes = 0.0
    total_overlap_nodes_frac = 0.0

    total_real_sg_edges = 0.0
    total_original_edges = 0.0

    count_subgraph_edges = 0
    count_subgraph_nodes = 0

    count_original_nodes = 0
    count_original_edges = 0

    for data in tqdm(dataset, desc="Computing subgraph statistics"):
        # each `data` is a SubgraphData corresponding to one original graph
        assert hasattr(data, "sg_nodes_orig"), "SubgraphData expected."
        assert hasattr(data, "sg_batch"), "SubgraphData expected."
        assert hasattr(data, "sg_graph"), "SubgraphData expected."

        total_graphs += 1

        N = int(data.num_nodes)          # original nodes in this graph
        E = int(data.edge_index.size(1)) # original edges in this graph

        P = int(data.num_subgraphs)      # number of subgraphs for this graph
        if P == 0 or N == 0 or E == 0:
            continue  # skip degenerate cases calmly

        total_subgraphs += P
        total_original_edges += E

        # --- subgraph sizes in nodes ---
        # sg_batch: subgraph id per subgraph-node copy
        sg_sizes = torch.bincount(data.sg_batch, minlength=P)  # [P]
        total_subgraph_nodes += float(sg_sizes.sum().item())
        total_subgraph_size_frac += float((sg_sizes.float() / N).sum().item())

        # --- overlapping nodes per subgraph ---
        # 1) count membership of each original node across subgraphs
        # data.sg_nodes_orig: [Tn_total] original node ids
        counts_per_node = torch.bincount(
            data.sg_nodes_orig,
            minlength=N
        )  # [N], how many subgraphs each original node appears in

        overlap_mask = counts_per_node >= 2  # original nodes that appear in >=2 subgraphs

        # 2) for each subgraph, count how many of its nodes are overlapping
        # map overlap_mask back to subgraph-node copies
        overlap_flag_per_copy = overlap_mask[data.sg_nodes_orig].float()  # [Tn_total]

        # sum per subgraph
        # we can do this with bincount over sg_batch
        overlap_per_sg = torch.bincount(
            data.sg_batch,
            weights=overlap_flag_per_copy,
            minlength=P
        )  # [P], float

        total_overlap_nodes += float(overlap_per_sg.sum().item())
        total_overlap_nodes_frac += float((overlap_per_sg / N).sum().item())

        # --- edges over subgraphs ---
        # "Real" subgraph edges are those with a corresponding original edge index.
        if hasattr(data, "sg_edges_orig"):
            real_edges_this_graph = int(data.sg_edges_orig.numel())
        else:
            real_edges_this_graph = 0

        total_real_sg_edges += float(real_edges_this_graph)

        # statistics per graph as a whole
        count_subgraph_edges += int(data.sg_edge_index.size(1))  # all subgraph edges (including added ones)
        count_subgraph_nodes += int(data.sg_nodes_orig.size(0))      # all subgraph nodes (including copies)
        count_original_nodes += N
        count_original_edges += E

    # avoid division by zero
    if total_graphs == 0 or total_subgraphs == 0 or total_original_edges == 0:
        return {
            "avg_subgraphs_per_graph": 0.0,
            "avg_subgraph_size_nodes": 0.0,
            "avg_subgraph_size_frac": 0.0,
            "avg_overlap_nodes_per_subgraph": 0.0,
            "avg_overlap_nodes_frac": 0.0,
            "avg_edges_per_subgraph": 0.0,
            "edges_ratio": 0.0,
        }

    stats = {
        # subgraphs
        "avg_subgraphs_per_graph": total_subgraphs / total_graphs,

        # subgraph sizes
        "avg_subgraph_size_nodes": total_subgraph_nodes / total_subgraphs,
        "avg_subgraph_size_frac": total_subgraph_size_frac / total_subgraphs,

        # node overlaps
        "avg_overlap_nodes_per_subgraph": total_overlap_nodes / total_subgraphs,
        "avg_overlap_nodes_frac": total_overlap_nodes_frac / total_subgraphs,

        # edges
        "avg_edges_per_subgraph": total_real_sg_edges / total_subgraphs,
        "edges_ratio": total_real_sg_edges / total_original_edges,

        # overall counts
        "graph_avg_count_subgraph_edges": count_subgraph_edges / total_graphs,
        "graph_avg_count_subgraph_nodes": count_subgraph_nodes / total_graphs,
        "graph_avg_count_original_nodes": count_original_nodes / total_graphs,
        "graph_avg_count_original_edges": count_original_edges / total_graphs,
    }

    return stats

# calculate dataset stats
def get_dataset_stats(example, cfg):
    """Infer dataset metadata from one example graph and the config.

    Args:
        example: Representative PyG ``Data`` object from the dataset.
        cfg: Experiment configuration containing dataset task and target
            settings.

    Returns:
        A dictionary describing feature dimensions, target structure, task type,
        and optional subgraph marking dimensions.
    """
    print("Calculating dataset stats...")
    stats = {}
    task_lower = cfg.dataset.task.lower()
    if getattr(example, "x", None) is None:
        raise ValueError(
            f"Dataset {cfg.dataset.name} does not provide node features. "
            "Add a node-feature pre-transform such as Constant."
        )
    stats['num_node_features'] = example.x.shape[-1] if example.x.ndim > 1 else 1
    if example.edge_attr is not None:
        stats['num_edge_features'] = example.edge_attr.shape[-1] if example.edge_attr.ndim > 1 else 1
    else:
        stats['num_edge_features'] = 0
    target_cols = cfg.dataset.get('target_column_names', None)
    if target_cols is not None:
        stats['target_column_names'] = list(OmegaConf.to_container(target_cols, resolve=True))
    else:
        stats['target_column_names'] = None
    if hasattr(example, "y") and example.y is not None:
        if example.y.ndim == 1:
            inferred_targets = example.y.shape[0]
        else:
            inferred_targets = example.y.shape[-1]
    else:
        inferred_targets = 1
    stats['num_targets'] = inferred_targets

    if 'multilabel' in task_lower or 'multi_label' in task_lower:
        stats['task_type'] = 'multilabel'
    elif "brec" in task_lower:
        stats['task_type'] = 'brec'
    elif 'classification' in task_lower:
        stats['task_type'] = 'classification'
    elif 'multi' in task_lower and 'regress' in task_lower:
        stats['task_type'] = 'multi_target_regression'
    else:
        stats['task_type'] = 'regression'

    if stats['task_type'] in {'classification', 'multilabel'}:
        stats['num_classes'] = cfg.dataset.num_classes
    elif stats['task_type'] == 'multi_target_regression':
        target_count = cfg.dataset.get('num_classes', inferred_targets)
        stats['num_classes'] = target_count
        stats['num_targets'] = target_count
    else:
        stats['num_classes'] = 1

    if stats['target_column_names'] is not None:
        stats['num_targets'] = len(stats['target_column_names'])
        if stats['task_type'] == 'multi_target_regression':
            stats['num_classes'] = len(stats['target_column_names'])
    stats['num_node_marks'] = example.sg_node_mark.shape[1] if hasattr(example, 'sg_node_mark') else 0
    stats["num_edge_marks"] = example.sg_edge_mark.shape[1] if hasattr(example, 'sg_edge_mark') else 0
    return stats

class FilterMinMaxNodes(object):
    def __init__(self, min_nodes, max_nodes):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def __call__(self, data: Data) -> bool:
        return self.min_nodes <= data.num_nodes <= self.max_nodes
    
class FilterOneComponentOnly(object):
    def __init__(self, use_weights: bool = False):
        self.use_weights = use_weights

    def __call__(self, data: Data) -> bool:
        """
        Return True only if the graph in `data` has exactly one connected component.
        We ignore edge weights for connectivity checks (use_weights kept for API compatibility).
        """
        if data.num_nodes == 0:
            return False

        # edge_index expected shape [2, E]
        edge_index = data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(int(data.num_nodes)))
        for src, dst in edge_index.T:
            G.add_edge(int(src), int(dst))

        num_components = nx.number_connected_components(G)
        return num_components == 1
    
class FilterNaNTarget(object):
    def __init__(self, target_name_in_dict: str = ""):
        print("Using target name from SelectLabel pre-transform: ", target_name_in_dict)
        self.get_label = SelectLabel(target=target_name_in_dict) if target_name_in_dict else None

    def __call__(self, data: Data) -> bool:
        """
        Return True only if the target label data.y is not NaN.
        """
        if self.get_label:
            label = self.get_label(data).y
        else:
            label = data.y

        if not torch.isfinite(label).all():
            print("Found non-finite label: ", label)
            return False
        return True

class ReplaceSelfLoops(BaseTransform):
    def __init__(self):
        super().__init__()
        
    def forward(self, data: Data) -> Data:
        self_loop_edge_index = torch.arange(0, data.num_nodes, 
                                            dtype=torch.long, 
                                            device=data.edge_index.device).unsqueeze(0).repeat(2, 1)
        data.edge_index = self_loop_edge_index
        if data.edge_attr is not None:
            data.edge_attr = torch.ones(data.num_nodes, data.edge_attr.size(1))
        return data


class SelectLabel(BaseTransform):
    def __init__(self, target: str):
        super().__init__()
        self.target = target

    def forward(self, data: Data) -> Data:
        """Select one target from ``data.y_dict`` and store it in ``data.y``.

        Args:
            data: Input graph with a ``y_dict`` mapping target names to labels.

        Returns:
            The same graph object with ``data.y`` set to the selected label.
        """
        if not hasattr(data, "y_dict"):
            raise AttributeError("Data object has no attribute 'y_dict'")
        labels = data.y_dict
        if self.target not in labels:
            raise ValueError(f"Target '{self.target}' not found in labels.")
        data.y = torch.tensor([labels[self.target]])
        return data


class LogTarget(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data) -> Data:
        """Apply a ``log1p`` transform to the target.

        Args:
            data: Input graph whose ``y`` attribute contains the target tensor.

        Returns:
            The same graph object with ``data.y`` transformed if present.
        """
        if not hasattr(data, "y") or data.y is None:
            return data
        data.y = (data.y + 1).log()
        assert torch.isfinite(data.y), "Transform introduces NaNs."
        return data
    
class AddIsConnected(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data) -> Data:
        """Attach a connectivity label to the graph.

        Args:
            data: Input graph.

        Returns:
            The same graph object with ``data.is_connected`` set to
            ``"connected"`` or ``"not_connected"``.
        """
        G = to_networkx(data, to_undirected=True)
        num_components = nx.number_connected_components(G)
        data.is_connected = "connected" if num_components == 1 else "not_connected"
        return data


class ExpandEdgeAttrs(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data) -> Data:
        data.edge_attr = data.edge_attr.unsqueeze(1)
        return data


class OneHotEdgeAttr(BaseTransform):
    def __init__(self, num_classes: int):
        """
        num_classes:
            - If given, uses this fixed number of edge types.
            - If None, infers num_classes = max(edge_attr) + 1 per graph.
              (For a dataset, it's usually better to pass a fixed num_classes.)
        """
        self.num_classes = num_classes
        self.highest_class = -1
        self.lowest_class = 1_000_000

    def forward(self, data: Data) -> Data:
        """One-hot encode a single categorical edge feature column.

        Args:
            data: Input graph whose ``edge_attr`` stores integer edge labels.

        Returns:
            The same graph object with one-hot encoded ``edge_attr`` when the
            input format is supported.
        """
        edge_attr = data.edge_attr
        if edge_attr.max().item() > self.highest_class:
            self.highest_class = edge_attr.max().item()
        if edge_attr.min().item() < self.lowest_class:
            self.lowest_class = edge_attr.min().item()

        if edge_attr is None:
            print("Warning: data has no edge_attr, skipping OneHotEdgeAttr transform.")
            return data
        
        if edge_attr.dtype != torch.long and edge_attr.dtype != torch.int64:
            print(f"Warning: edge_attr dtype is {edge_attr.dtype}, expected integer types. Skipping OneHotEdgeAttr transform.")
            return data

        # Allow shapes [E] or [E, 1] of integer edge types
        if edge_attr.dim() == 2 and edge_attr.size(-1) == 1:
            edge_attr = edge_attr.squeeze(-1)

        assert edge_attr.dim() == 1, (
            f"Expected edge_attr to be 1D integer tensor of edge types, "
            f"got shape {edge_attr.shape}, "
            f"dtype {edge_attr.dtype}"
            f"prinout edge_attr: {edge_attr}"
        )
        data.edge_attr = F.one_hot(edge_attr.long(), num_classes=self.num_classes).to(torch.float)
        assert self.highest_class + 1 <= self.num_classes, (
            f"num_classes {self.num_classes} is not sufficient for edge types "
            f"in the data (found edge types in range [{self.lowest_class}, {self.highest_class}])"
        )
        return data
    

class OneHotNodeFeatures(BaseTransform):
    def __init__(self, num_classes: int):
        """
        num_classes:
            - If given, uses this fixed number of edge types.
            - If None, infers num_classes = max(edge_attr) + 1 per graph.
              (For a dataset, it's usually better to pass a fixed num_classes.)
        """
        self.num_classes = num_classes

    def forward(self, data: Data) -> Data:
        """One-hot encode a single categorical node feature column.

        Args:
            data: Input graph whose ``x`` stores integer node labels.

        Returns:
            The same graph object with one-hot encoded ``x`` when the input
            format is supported.
        """
        node_feat = data.x
        if node_feat is None:
            print("Warning: data has no node features, skipping OneHotNodeFeatures transform.")
            return data
        
        if node_feat.dtype != torch.long and node_feat.dtype != torch.int64:
            print(f"Warning: node feature dtype is {node_feat.dtype}, expected integer types. Skipping OneHotNodeFeatures transform.")
            return data
        
        if node_feat.dim() == 2 and node_feat.size(-1) == 1:
            node_feat = node_feat.squeeze(-1)
            
        assert node_feat.dim() == 1, (
            f"Expected node features to be 1D integer tensor of node types, "
            f"got shape {node_feat.shape}, "
            f"dtype {node_feat.dtype}"
        )
        data.x = F.one_hot(node_feat.long(), num_classes=self.num_classes).to(torch.float)
        return data


class MultiColumnOneHotEdgeAttr(BaseTransform):
    def __init__(self, num_classes_per_col: list[int]):
        """
        num_classes_per_col:
            A list/tuple with one entry per edge_attr column.
            Example:
              - ZINC (1 int column):        [4]
              - Peptides-func (3 columns):  [n0, n1, n2]
        """
        self.num_classes_per_col = list(num_classes_per_col)
        self.highest_class_per_col = [-1] * len(self.num_classes_per_col)
        self.lowest_class_per_col = [1_000_000_000] * len(self.num_classes_per_col)

    def forward(self, data: Data) -> Data:
        """One-hot encode each categorical edge-feature column separately.

        Args:
            data: Input graph whose ``edge_attr`` has shape ``[E, C]`` with
                integer-coded categories per column.

        Returns:
            The same graph object with concatenated one-hot encoded
            ``edge_attr``.
        """
        edge_attr = data.edge_attr

        # keep track of min and max per colunn
        for col_idx in range(len(self.num_classes_per_col)):
            col = edge_attr[:, col_idx]
            if col.max().item() > self.highest_class_per_col[col_idx]:
                self.highest_class_per_col[col_idx] = col.max().item()
            if col.min().item() < self.lowest_class_per_col[col_idx]:
                self.lowest_class_per_col[col_idx] = col.min().item()

        # No edges / no edge features → nothing to do
        if edge_attr is None:
            print("Warning: data has no edge_attr, skipping MultiColumnOneHotEdgeAttr transform.")
            return data

        if edge_attr.dtype not in (torch.long, torch.int64):
            print(f"Warning: edge_attr dtype is {edge_attr.dtype}, expected integer types. Skipping MultiColumnOneHotEdgeAttr transform.")
            return data

        assert edge_attr.dim() == 2, (
            f"Expected edge_attr to be 2D tensor of shape [E, C], "
            f"got shape {edge_attr.shape}, "
            f"dtype {edge_attr.dtype}"
        )

        E, C = edge_attr.size()
        assert C == len(self.num_classes_per_col), \
            f"edge_attr has {C} columns, but num_classes_per_col has {len(self.num_classes_per_col)} entries"

        onehots = []
        for col_idx, num_classes in enumerate(self.num_classes_per_col):
            col = edge_attr[:, col_idx]          # [E]
            onehots.append(
                F.one_hot(col, num_classes=num_classes).float()
            )  # [E, num_classes]

        data.edge_attr = torch.cat(onehots, dim=-1)  # [E, sum(num_classes_per_col)]

        # print lowest and highest per column so far
        for col_idx, num_classes in enumerate(self.num_classes_per_col):
            assert self.lowest_class_per_col[col_idx] <= num_classes - 1 <= self.highest_class_per_col[col_idx], (
                f"num_classes {num_classes} for column {col_idx} is not sufficient for edge types "
                f"in the data (found edge types in range [{self.lowest_class_per_col[col_idx]}, {self.highest_class_per_col[col_idx]}])"
            )
        return data

def fetch_transform(cfg):
    """Construct configured pre-filters, pre-transforms, and runtime transforms.

    Args:
        cfg: Experiment configuration containing dataset transform and filter
            specifications.

    Returns:
        A tuple ``(pre_transform, pre_filter, transform)`` of composed PyG
        transform objects or ``None`` when a stage is unused.
    """

    # empty transforms
    transforms = []
    pre_transforms = []
    pre_filters = []

    if cfg.dataset.pre_filter:
        for f in cfg.dataset.pre_filter:
            if f['name'] == 'FilterNodes':
                min_nodes = f['params'].get('min_nodes', 0)
                max_nodes = f['params'].get('max_nodes', 1_000_000_000)
                pre_filters.append(FilterMinMaxNodes(min_nodes=min_nodes, max_nodes=max_nodes))
            elif f['name'] == 'FilterOneComponentOnly':
                pre_filters.append(FilterOneComponentOnly())
            elif f['name'] == 'FilterNaNTarget':
                target_name = ""
                # check if select label exists in the pre transform list
                if any(t['name'] == 'SelectLabel' for t in cfg.dataset.pre_transform):
                    # get target name from select label
                    target_name = next(t['params']['target'] for t in cfg.dataset.pre_transform if t['name'] == 'SelectLabel')
                pre_filters.append(FilterNaNTarget(target_name_in_dict=target_name))
            else:
                raise ValueError(f"Unknown pre-filter: {f['name']}")
    
     # pre-transforms
    if cfg.dataset.pre_transform:
        for t in cfg.dataset.pre_transform:
            if t['name'] == 'ToUndirected':
                pre_transforms.append(T.ToUndirected())
            elif t['name'] == 'ReplaceSelfLoops':
                pre_transforms.append(ReplaceSelfLoops())
            elif t['name'] == 'SelectLabel':
                pre_transforms.append(SelectLabel(target=t['params']['target']))
            elif t['name'] == 'LogTarget':
                pre_transforms.append(LogTarget())
            elif t['name'] == 'Constant':
                pre_transforms.append(T.Constant())
            elif t['name'] == 'AddIsConnected':
                pre_transforms.append(AddIsConnected())
            elif t['name'] == 'ExpandEdgeAttrs':
                pre_transforms.append(ExpandEdgeAttrs())
            elif t['name'] == 'OneHotEdgeAttr':
                num_classes = t['params'].get('num_classes', None)
                pre_transforms.append(OneHotEdgeAttr(num_classes=num_classes))
            elif t['name'] == 'MultiColumnOneHotEdgeAttr':
                num_classes_per_col = t['params']['num_classes_per_col']
                pre_transforms.append(MultiColumnOneHotEdgeAttr(num_classes_per_col=num_classes_per_col))
            elif t['name'] == 'OneHotNodeFeatures':
                num_classes = t['params'].get('num_classes', None)
                pre_transforms.append(OneHotNodeFeatures(num_classes=num_classes))
            else:
                raise ValueError(f"Unknown pre-transform: {t['name']}")

    if (
        cfg.model.name == "mp_gnn"
        and cfg.model.conv_type == "GPS"
        and cfg.model.get("use_pe_for_transformers", False)
    ):
        pe_type = str(cfg.model.get("pe_type", "rwse")).lower()
        pe_dim = cfg.model.get("pe_dim", cfg.model.get("rwse_dim", 16))
        if pe_type == "rwse":
            pre_transforms.append(
                T.AddRandomWalkPE(
                    walk_length=pe_dim,
                    attr_name=None,
                )
            )
        elif pe_type == "laplacian":
            pre_transforms.append(
                T.AddLaplacianEigenvectorPE(
                    k=pe_dim,
                    attr_name=None,
                    is_undirected=True,
                )
            )
        else:
            raise ValueError(f"Unsupported GPS positional encoding type: {pe_type}")

    if cfg.dataset.transform:
        raise NotImplementedError("No transforms implemented yet.")

    # compose transforms
    pre_transform = None
    if len(pre_transforms) > 0:
        pre_transform = Compose(pre_transforms)
    
    pre_filter = None
    if len(pre_filters) > 0:
        pre_filter = ComposeFilters(pre_filters)
    
    transform = None
    if len(transforms) > 0:
        transform = Compose(transforms)

    print("FINAL PRE-TRANSFORM:", pre_transform)
    print("FINAL PRE-FILTER:", pre_filter)
    print("FINAL TRANSFORM:", transform)

    return pre_transform, pre_filter, transform
