from torch import nn
import torch
from torch_geometric.nn import GCNConv, GATConv, GINEConv, GINConv, GPSConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import GraphNorm

def conv_getter(
    conv_type,
    in_channels,
    out_channels,
    p_drop=0.,
    use_norm=True,
    gps_attn_type="performer",
):
    """Instantiate a graph convolution layer by name.

    Args:
        conv_type: Convolution family identifier such as ``GCN``, ``GIN``, or
            ``GINE``.
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        p_drop: Dropout applied inside MLP-style convolution blocks.
        use_norm: Whether to include normalization inside applicable blocks.
        gps_attn_type: Attention backend for GPSConv, e.g. ``performer`` or
            ``multihead``.

    Returns:
        A configured PyG convolution module.
    """
    chosen_norm = nn.BatchNorm1d(out_channels) if use_norm else nn.Identity()
    if conv_type == 'GCN':
        return GCNConv(in_channels, out_channels)
    elif conv_type == 'GIN':
        nn_ = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            chosen_norm,
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(p_drop),
        )
        return GINConv(nn_)
    elif conv_type == 'GINE':
        nn_ = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            chosen_norm,
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(p_drop),
        )
        return GINEConv(nn_, edge_dim=out_channels)
    elif conv_type == 'GAT':
        return GATConv(in_channels, out_channels // 8, heads=8)
    elif conv_type == "GPS":
        mp_gnn = GCNConv(in_channels, in_channels)
        conv = GPSConv(out_channels, 
                       conv=mp_gnn, 
                       heads=2, 
                       dropout=p_drop,
                       attn_type=gps_attn_type)
        return conv
    else:
        raise ValueError(f"Unsupported conv_type: {conv_type}")

class FFN_Layer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, p_drop=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(p_drop),
        )
        self.bn_out = nn.BatchNorm1d(out_dim)
    
    def forward(self, x):
        return self.bn_out(x + self.ffn(x))


class MLP_Head(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            # layer 1
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            # layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            # classifier
            nn.Linear(hidden_dim, out_dim),
                        )
    
    def forward(self, x):
        return self.net(x)


class BaseModel(nn.Module):
    """Shared base class for graph and subgraph models in this repository.

    Args:
        cfg: Experiment configuration with dataset and model hyperparameters.
        data_stats: Dataset-derived feature and target metadata.

    Returns:
        A model base object with common encoders, pooling, and helper methods
        used by concrete model classes.
    """
    def __init__(self, 
                 cfg,
                 data_stats):
        super().__init__()


        ## read in config file and set relevant attributes

        # incoming data
        self.dataset_name = cfg.dataset.name
        self.in_dim = data_stats["num_node_features"]
        self.edge_dim = data_stats["num_edge_features"]
        self.node_mark_dim = data_stats["num_node_marks"] if "num_node_marks" in data_stats else 0
        self.edge_mark_dim = data_stats["num_edge_marks"] if "num_edge_marks" in data_stats else 0
        self.out_dim = data_stats["num_classes"] if "num_classes" in data_stats else 1
        # model params
        self.embedding_dim = cfg.dataset.model_params.hidden_dim
        self.num_layers = cfg.dataset.model_params.num_layers
        self.final_pooling = cfg.dataset.model_params.readout
        self.head_dropout = cfg.dataset.model_params.head_dropout
        self.conv_type = cfg.model.conv_type
        self.conv_dropout = cfg.dataset.model_params.conv_dropout
        self.interlayer_mlp = cfg.dataset.model_params.intermediate_mlp
        self.use_subgraph_mlp = cfg.model.get("use_subgraph_mlp", False)

        self.within_conv_dropout = 0.0 # NOTE: set to 0 for now as our models include seperate dropouts in each loop over layers

        # encoding strategy
        self.encoding_strategy = cfg.model.encoding_strategy

        # compute initial embeddings sizes accounting for markings
        self.initial_node_embed_dim = self.embedding_dim - self.node_mark_dim if self.encoding_strategy == "CONCAT" else self.embedding_dim
        self.initial_edge_embed_dim = self.embedding_dim - self.edge_mark_dim if self.encoding_strategy == "CONCAT" else self.embedding_dim

        ## instantiate dataset-specific feature and edge encoder
        if "zinc" in self.dataset_name.lower():
            print("Using ZINC dataset node and edge encoders.")
            self.node_encoder = nn.Embedding(28, embedding_dim=self.initial_node_embed_dim)
            self.edge_encoder = nn.Embedding(4, embedding_dim=self.initial_edge_embed_dim)
        else:
            self.node_encoder = nn.Linear(self.in_dim, self.initial_node_embed_dim)
            if self.edge_dim is not None and self.edge_dim > 0:
                self.edge_encoder = nn.Linear(self.edge_dim, self.initial_edge_embed_dim)
            else:
                self.edge_encoder = None

        if self.node_mark_dim > 0 and self.encoding_strategy == "SUM":
            self.node_mark_encoder = nn.Linear(self.node_mark_dim, self.initial_node_embed_dim)

        ### instantiate final pooling layer
        if self.final_pooling == "sum":
            self.pool = global_add_pool
        elif self.final_pooling == "mean":
            self.pool = global_mean_pool
        elif self.final_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling type: {self.final_pooling}")
        
    def perform_conv(self, conv, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            return conv(x, edge_index, edge_attr), edge_attr
        else:
            return conv(x, edge_index), None
        
    def graph_level_pooler(self, x, batch):
        return self.pool(x, batch)
    
    def embed_node_edges(self, x, edge_attr):
        """Encode raw node and edge features into the model embedding space.

        Args:
            x: Raw node feature tensor.
            edge_attr: Raw edge feature tensor or ``None``.

        Returns:
            A tuple ``(x, edge_attr)`` of embedded node and edge features.
        """
        # embed node features
        if self.node_encoder is not None:
            x = x.squeeze() if x.dim() == 3 else x
            if "zinc" in self.dataset_name.lower():
                x = x.squeeze().long()
            x = self.node_encoder(x)

        # embed edge features
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = edge_attr.squeeze() if edge_attr.dim() == 3 else edge_attr
            if "zinc" in self.dataset_name.lower():
                edge_attr = edge_attr.squeeze().long()
            edge_attr = self.edge_encoder(edge_attr)

        return x, edge_attr
    
    def forward(self, data):
        raise NotImplementedError("Forward method not implemented in BaseModel.")
    
    def create_layer_conv(self):
        return conv_getter(self.conv_type, 
                           self.embedding_dim, 
                           self.embedding_dim, 
                           self.within_conv_dropout,
                           use_norm=(self.dataset_name!="BREC"))  # use batchnorm in GIN except for BREC
    
    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _build_subgraph_node_features(self, batch):
            """Construct node features in subgraph-node space.

            Args:
                batch: ``SubgraphData`` object with original node features and
                    subgraph membership tensors.

            Returns:
                Node features indexed by subgraph-node copies, optionally
                combined with node markings.
            """
            x = batch.x  # [N_total, F]
            x_sg = x[batch.sg_nodes_orig] # [Tn_total, F]

            if self.encoding_strategy == "CONCAT":
                if hasattr(batch, 'sg_node_mark'):
                    x_sg = torch.cat([x_sg, batch.sg_node_mark], dim=-1)
            elif self.encoding_strategy == "SUM":
                if hasattr(batch, 'sg_node_mark'):
                    mark_enc = self.node_mark_encoder(batch.sg_node_mark)
                    x_sg = x_sg + mark_enc
            else:
                raise ValueError(f"Unsupported encoding strategy: {self.encoding_strategy}")

            return x_sg.contiguous()  # [Tn_total, in_channels + mark_dim]

    def _build_subgraph_edge_attr(self, batch):
        """Construct edge features for subgraph edges.

        Args:
            batch: ``SubgraphData`` object with original edge features and
                optional subgraph edge markings.

        Returns:
            Edge features aligned with ``batch.sg_edge_index``, or ``None`` if
            no edge-level features are available.
        """
        num_total_edges = batch.sg_edge_index.size(1)
        parts = []

        # base edge features from original graph
        base = None
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            if hasattr(batch, 'sg_edges_orig'):
                base = batch.edge_attr[batch.sg_edges_orig]  # [Te_real, edge_dim]
                if base.size(0) < num_total_edges:
                    pad = base.new_zeros(num_total_edges - base.size(0), base.size(1))
                    base = torch.cat([base, pad], dim=0)
                parts.append(base)

        # edge markings (already in subgraph-edge space, size Te_real x edge_mark_dim)
        if hasattr(batch, 'sg_edge_mark'):
            marks = batch.sg_edge_mark
            if marks.size(0) < num_total_edges:
                pad = marks.new_zeros(num_total_edges - marks.size(0), marks.size(1))
                marks = torch.cat([marks, pad], dim=0)
            parts.append(marks)

        if not parts:
            return None
        return torch.cat(parts, dim=-1).contiguous()  # [Te_total, total_edge_in_dim]
