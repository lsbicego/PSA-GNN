import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool
import torch_scatter
from .layers import MLP_Head, conv_getter, BaseModel, FFN_Layer
from torch_geometric.utils import to_dense_batch
import torch

from torch_geometric.nn.norm import GraphNorm

# =============================================================================
# 1. Subgraph Layers
# =============================================================================

class KernelSubgraphLayer(nn.Module):
    """Subgraph layer that mixes within-subgraph updates with pooled summaries.

    Args:
        in_dim: Input node embedding size.
        out_dim: Output node embedding size.
        inner_layers: Number of within-subgraph message-passing layers.
        conv_type: Convolution family used inside the subgraph.
        within_conv_dropout: Dropout applied inside the convolution blocks.

    Returns:
        A layer that updates subgraph-node features using within-subgraph,
        per-subgraph, and per-original-node information.
    """
    def __init__(self, in_dim, out_dim, inner_layers: int=2, conv_type='GIN', within_conv_dropout=0.):
        super().__init__()
        self.conv_type = conv_type
        
        self.convs = nn.ModuleList()
        for i, layer in enumerate(range(inner_layers)):
            if layer == 0:
                in_channels = in_dim
            else:
                in_channels = out_dim
            conv = conv_getter(conv_type, in_channels, out_dim, p_drop=within_conv_dropout)
            self.convs.append(conv)

        self.subgraph_x_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.duplicated_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )


    def forward(self, sg_x, sg_edge_attr, batch):
        # Apply the within-subgraph message-passing stack.
        node_x = sg_x
        for conv in self.convs:
            if self.conv_type == 'GINE' and sg_edge_attr is not None:
                node_x = conv(node_x, batch.sg_edge_index, sg_edge_attr)
            else:
                node_x = conv(node_x, batch.sg_edge_index)
            node_x = F.relu(node_x)

        subgraph_x = self.subgraph_x_mlp(global_mean_pool(node_x, batch.sg_batch))  # [P_total, hidden]

        # Aggregate repeated node copies back to the original graph.
        duplicated_x = self.duplicated_mlp(torch_scatter.scatter_add(
            node_x,
            batch.sg_nodes_orig,
            dim=0,
            dim_size=batch.num_nodes,
        )  # [N_total, hidden]
        )
        
        x = self.node_mlp(node_x) + duplicated_x[batch.sg_nodes_orig] + subgraph_x[batch.sg_batch]

        return x

class AblationLayer(nn.Module):
    """Configurable subgraph layer used for ablation studies.

    Args:
        in_dim: Input node embedding size.
        out_dim: Output node embedding size.
        conv_type: Convolution family for local and global updates.
        within_conv_dropout: Dropout applied in the convolution blocks.
        add_subgraph_repr: Whether to inject a pooled subgraph representation.
        duplicated_sum_mean: Reduction used for merging duplicate node copies.
        add_mlps: Whether to post-process branches with small MLPs.

    Returns:
        A layer that combines local subgraph updates with aggregated global and
        optional subgraph-level features.
    """
    def __init__(self, in_dim, out_dim, 
                 conv_type='GIN', 
                 within_conv_dropout=0.,
                 add_subgraph_repr: bool=False,
                 duplicated_sum_mean: str='mean', # 'sum' or 'mean'
                 add_mlps: bool=False,
                 ):
        super().__init__()
        self.conv_type = conv_type
        
        self.sg_conv = conv_getter(conv_type, in_dim, out_dim, p_drop=within_conv_dropout)
        self.global_conv = conv_getter(conv_type, out_dim, out_dim, p_drop=within_conv_dropout)

        self.add_mlps = add_mlps
        self.add_subgraph_repr = add_subgraph_repr
        self.duplicated_sum_mean = duplicated_sum_mean

        if self.add_mlps:
            def _make_mlp():
                return nn.Sequential(
                    nn.Linear(out_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(out_dim, out_dim),
                )
            self.sg_node_mlp = _make_mlp()
            self.duplicated_mlp = _make_mlp()
            self.subgraph_mlp = _make_mlp()
        
        if self.duplicated_sum_mean == 'sum':
            self.dup_aggregator = torch_scatter.scatter_add
        elif self.duplicated_sum_mean == 'mean':
            self.dup_aggregator = torch_scatter.scatter_mean
        


    def forward(self, sg_x, sg_edge_attr, batch):

        if self.conv_type == 'GINE' and sg_edge_attr is not None:
            node_x = self.sg_conv(sg_x, batch.sg_edge_index, sg_edge_attr)
        else:
            node_x = self.sg_conv(sg_x, batch.sg_edge_index)

        global_x = self.dup_aggregator(
            node_x,
            batch.sg_nodes_orig,
            dim=0,
            dim_size=batch.num_nodes,
        )  # [N_total, hidden]

        if self.conv_type == 'GINE' and batch.edge_attr is not None:
            global_x = self.global_conv(global_x, batch.edge_index, batch.edge_attr)
        else:
            global_x = self.global_conv(global_x, batch.edge_index)

        if self.add_subgraph_repr:
            subgraph_x = self.dup_aggregator(
                node_x,
                batch.sg_batch,
                dim=0,
                dim_size=batch.num_subgraphs,
            )  # [P_total, hidden]
        else:
            subgraph_x = torch.zeros((batch.num_subgraphs, node_x.size(1)), device=node_x.device)
        
        if self.add_mlps:
            node_x = self.sg_node_mlp(node_x)
            global_x = self.duplicated_mlp(global_x)
            subgraph_x = self.subgraph_mlp(subgraph_x)
        
        return node_x + global_x[batch.sg_nodes_orig] + subgraph_x[batch.sg_batch]

class PSALayer(nn.Module):
    """PSA-style layer over explicit subgraph-node copies.

    Args:
        in_dim: Input node embedding size.
        out_dim: Output node embedding size.
        conv_type: Convolution family used for local and global updates.
        within_conv_dropout: Dropout applied inside the convolution blocks.
        aggregation: Reduction used when merging duplicate node copies.

    Returns:
        A layer that updates subgraph-node features using within-subgraph
        message passing plus an aggregated original-graph branch.
    """
    def __init__(self, in_dim, out_dim, conv_type='GIN', within_conv_dropout=0., aggregation="mean"):
        super().__init__()
        self.conv_type = conv_type
        
        self.sg_conv = conv_getter(conv_type, in_dim, out_dim, p_drop=within_conv_dropout)
        self.sg_bn = nn.BatchNorm1d(out_dim)

        self.global_conv = conv_getter(conv_type, out_dim, out_dim, p_drop=within_conv_dropout)
        self.global_bn = nn.BatchNorm1d(out_dim)     

        self.agg_func = torch_scatter.scatter_mean if aggregation == "mean" else torch_scatter.scatter_add   

    def forward(self, sg_x, sg_edge_attr, batch):

        if self.conv_type == 'GINE' and sg_edge_attr is not None:
            node_x = self.sg_conv(sg_x, batch.sg_edge_index, sg_edge_attr)
        else:
            node_x = self.sg_conv(sg_x, batch.sg_edge_index)

        # Aggregate subgraph-node copies back to original nodes.
        global_x = self.agg_func(
            sg_x,
            batch.sg_nodes_orig,
            dim=0,
            dim_size=batch.num_nodes,
        )  # [N_total, hidden]

        if self.conv_type == 'GINE' and batch.edge_attr is not None:
            global_x = self.global_conv(global_x, batch.edge_index, batch.edge_attr)
        else:
            global_x = self.global_conv(global_x, batch.edge_index)

        node_x += global_x[batch.sg_nodes_orig]
        return node_x


class SUNLayer(nn.Module):
    """SUN-style layer adapted to the repository's subgraph batch format.

    Args:
        emb_dim: Node embedding size.
        conv_type: Convolution family used in the local and global branches.
        use_readout: Whether to inject pooled subgraph information.
        subgraph_readout: Reduction used for subgraph-level pooling.
        add_bn: Whether to apply batch normalization in intermediate branches.
        use_mlp: Whether to use MLP projections instead of linear projections.
        within_conv_dropout: Dropout applied in the convolution blocks.

    Returns:
        A SUN-style update over subgraph-node copies that mixes local, root, and
        original-graph information.
    """
    def __init__(
        self,
        emb_dim: int,
        conv_type: str = "GIN",
        *,
        use_readout: bool = True,
        subgraph_readout: str = "mean",  # "sum" or "mean"
        add_bn: bool = True,
        use_mlp: bool = True,
        within_conv_dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.conv_type = conv_type
        self.use_readout = use_readout

        # subgraph readout pool
        if subgraph_readout == "sum":
            self.sg_pool = global_add_pool
        elif subgraph_readout == "mean":
            self.sg_pool = global_mean_pool
        else:
            raise ValueError("subgraph_readout must be 'sum' or 'mean'")

        # convs: union-of-subgraphs stream + original-graph stream
        self.gnn = conv_getter(conv_type, emb_dim, emb_dim, p_drop=within_conv_dropout)
        self.gnn_sum = conv_getter(conv_type, emb_dim, emb_dim, p_drop=within_conv_dropout)

        # separate root convs (matches official SUN structure)
        self.gnn_root = conv_getter(conv_type, emb_dim, emb_dim, p_drop=within_conv_dropout)
        self.gnn_root_sum = conv_getter(conv_type, emb_dim, emb_dim, p_drop=within_conv_dropout)

        # norms
        self.bn = nn.BatchNorm1d(emb_dim) if add_bn else nn.Identity()
        self.bn_sum = nn.BatchNorm1d(emb_dim) if add_bn else nn.Identity()
        self.bn_root = nn.BatchNorm1d(emb_dim) if add_bn else nn.Identity()
        self.bn_root_sum = nn.BatchNorm1d(emb_dim) if add_bn else nn.Identity()

        # u_* transforms
        def make_u():
            if use_mlp:
                return nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                )
            else:
                return nn.Linear(emb_dim, emb_dim)

        self.u_readout = make_u()
        self.u_readout_root = make_u()

        self.u_vv = make_u()
        self.u_kk = make_u()
        self.u_kv = make_u()
        self.u_vv_root = make_u()

    @staticmethod
    def _build_root_occ_per_subgraph(batch, x):
        """
        Build mapping: subgraph_id -> root occurrence index into x.
        Requires exactly one root occurrence per subgraph.
        """
        is_root = batch.sg_is_root.bool().squeeze()                       # [Tn_total]
        root_idx = is_root.nonzero(as_tuple=False).reshape(-1)                # [num_roots]
        root_sg = batch.sg_batch[root_idx]                                                  # [num_roots]

        # Expect exactly one root per subgraph:
        # (If you want to allow missing roots, you can relax this and handle defaults.)
        root_occ_per_sg = torch.empty(batch.num_subgraphs, device=x.device, dtype=torch.long)
        root_occ_per_sg[root_sg] = root_idx

        return is_root, root_idx, root_sg, root_occ_per_sg

    def forward(self, x, sg_edge_attr, batch):
        """
        Returns raw out (no relu/dropout), so your outer loop can decide.
        """
        node_idx = batch.sg_nodes_orig   # v: [Tn_total]
        sg_idx = batch.sg_batch          # subgraph id: [Tn_total]

        # ---- roots ----
        is_root, root_idx, root_sg, root_occ_per_sg = self._build_root_occ_per_subgraph(batch, x)

        # For each occurrence i, get its subgraph's root occurrence
        root_occ_for_i = root_occ_per_sg[sg_idx]                         # [Tn_total]
        root_repr_for_i = x[root_occ_for_i]                              # [Tn_total, d]

        # ---- build root_repr_per_node + mask (ACL-compatible) ----
        # root_orig: original node id of each root occurrence (at most once per node is ideal)
        root_orig = batch.sg_nodes_orig[root_idx]                         # [num_roots]
        root_repr = x[root_idx]                                           # [num_roots, d]

        has_root = torch.zeros(batch.num_nodes, device=x.device, dtype=torch.bool)
        has_root[root_orig] = True

        root_repr_per_node = torch.zeros(batch.num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        root_repr_per_node[root_orig] = root_repr

        # ---- subgraph readout (optional) ----
        # Pool occurrences -> subgraph representations, then broadcast back
        if self.use_readout:
            h_subgraph = self.sg_pool(x, sg_idx)                          # [num_subgraphs, d]
            readout = self.u_readout(h_subgraph)                          # [num_subgraphs, d]
            readout_root = self.u_readout_root(h_subgraph)                # [num_subgraphs, d]
        else:
            readout = None
            readout_root = None

        # ---- collapsed original-node features for h2 ----
        x_sum = torch_scatter.scatter_mean(
            src=x,
            index=node_idx,
            dim=0,
            dim_size=batch.num_nodes,
        )                                                                 # [num_nodes, d]

        # ---- h1: union-of-subgraphs conv ----
        if self.conv_type == "GINE" and sg_edge_attr is not None:
            h1_all = self.gnn(x, batch.sg_edge_index, sg_edge_attr)
            h1_root_all = self.gnn_root(x, batch.sg_edge_index, sg_edge_attr)
        else:
            h1_all = self.gnn(x, batch.sg_edge_index)
            h1_root_all = self.gnn_root(x, batch.sg_edge_index)

        # match official: BN on non-roots separately
        h1 = h1_all.clone()
        if (~is_root).any():
            h1[~is_root] = self.bn(h1[~is_root])

        # root branch
        h1_root = self.bn_root(h1_root_all[is_root])                      # [num_roots, d]

        # ---- h2: original-graph conv on x_sum ----
        if self.conv_type == "GINE" and getattr(batch, "edge_attr", None) is not None:
            h2 = self.bn_sum(self.gnn_sum(x_sum, batch.edge_index, batch.edge_attr))
            h2_root = self.bn_root_sum(self.gnn_root_sum(x_sum, batch.edge_index, batch.edge_attr))
        else:
            h2 = self.bn_sum(self.gnn_sum(x_sum, batch.edge_index))
            h2_root = self.bn_root_sum(self.gnn_root_sum(x_sum, batch.edge_index))

        # ---- needle terms (no transpose) ----
        v = node_idx                                                     # [Tn_total]

        # x_kk: root of subgraph broadcast within that subgraph
        x_kk = self.u_kk(root_repr_for_i)                                 # [Tn_total, d]

        # x_kv: node-local
        x_kv = self.u_kv(x)                                               # [Tn_total, d]

        # x_vv: root(v) broadcast to occurrences of v; masked if v has no root
        x_vv_raw = self.u_vv(root_repr_per_node)[v]                       # [Tn_total, d]
        x_vv = x_vv_raw * has_root[v].unsqueeze(-1)                       # [Tn_total, d]

        out = x_vv + x_kk + x_kv + h1 + h2[v]                             # [Tn_total, d]
        if self.use_readout:
            out = out + readout[sg_idx]

        # ---- root override/update ----
        root_repr_occ = x[is_root]                                        # [num_roots, d]
        out_root = self.u_vv_root(root_repr_occ) + h1_root + h2_root[node_idx[is_root]]
        if self.use_readout:
            out_root = out_root + readout_root[sg_idx[is_root]]

        out[is_root] = out_root
        return out
    

class DSLayer(nn.Module):
    def __init__(self, in_dim, out_dim, conv_type='GIN', within_conv_dropout=0.):
        super().__init__()
        self.conv_type = conv_type
        self.sg_conv = conv_getter(conv_type, in_dim, out_dim, p_drop=within_conv_dropout)
        
    def forward(self, sg_x, sg_edge_attr, batch):
        if self.conv_type == 'GINE' and sg_edge_attr is not None:
            node_x = self.sg_conv(sg_x, batch.sg_edge_index, sg_edge_attr)
        else:
            node_x = self.sg_conv(sg_x, batch.sg_edge_index)

        # output is simply node representation after single convolution
        x = node_x
        return x


class HierarchicalSubgraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, conv_type='GIN', within_conv_dropout=0.):
        super().__init__()

        self.edge_weight_encoder = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.ReLU(inplace=True),
        )

        # global conv
        self.global_conv = conv_getter(conv_type, in_dim, out_dim, p_drop=within_conv_dropout)
        self.global_bn = nn.BatchNorm1d(out_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            )

        # subgraph conv
        self.subgraph_conv = conv_getter(conv_type, out_dim, out_dim, p_drop=within_conv_dropout)
        self.subgraph_bn = nn.BatchNorm1d(out_dim)
        self.subgraph_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            )


        # clustered conv 
        self.clustered_conv = conv_getter("GINE", out_dim, out_dim, p_drop=within_conv_dropout) # always GINE here since we have edge weights
        self.clustered_bn = nn.BatchNorm1d(out_dim)
        self.clustered_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            )


    def forward(self, sg_x, sg_edge_attr, batch):
    
        # Subgraph-level update.
        x_subgraph = self.subgraph_conv(sg_x, batch.sg_edge_index, sg_edge_attr) 
        x_subgraph = self.subgraph_bn(x_subgraph)
        x_subgraph = F.relu(x_subgraph)
        x_subgraph = self.subgraph_mlp(x_subgraph)

        # Original-graph update from aggregated subgraph-node copies.
        x_global = torch_scatter.scatter_add(sg_x, batch.sg_nodes_orig, dim=0, dim_size=batch.num_nodes)  # [N_total, hidden]

        # Recover one edge feature per original edge from repeated subgraph copies.
        if sg_edge_attr is not None:
            sg_edge_attr_expanded = sg_edge_attr[batch.sg_edges_orig].contiguous()  # [E_total, edge_dim]
            edge_attr = torch_scatter.scatter_mean(
                sg_edge_attr_expanded,
                batch.sg_edges_orig,
                dim=0,
                dim_size=batch.num_edges,
            )  # [E_total_unique, edge_dim]
        else:
            edge_attr = None

        x_global = self.global_conv(x_global, batch.edge_index, edge_attr)
        x_global = self.global_bn(x_global)
        x_global = F.relu(x_global)
        x_global = self.global_mlp(x_global)
        x_global = x_global[batch.sg_nodes_orig].contiguous() # collapse back to subgraph nodes

        # 3) clustered conv
        x_clustered = torch_scatter.scatter_add(sg_x, batch.sg_batch, dim=0, dim_size=batch.num_subgraphs)  # [P_total, hidden]

        # NOTE: this uses edge weights = number of edges between super-nodes in original graph
        edge_weight_encoded = self.edge_weight_encoder(batch.super_edge_weight.unsqueeze(-1))  # [S_total, hidden]

        x_clustered = self.clustered_conv(x_clustered, batch.super_edge_index, edge_weight_encoded)
        x_clustered = self.clustered_bn(x_clustered)
        x_clustered = F.relu(x_clustered)
        x_clustered = self.clustered_mlp(x_clustered)
        x_clustered = x_clustered[batch.sg_batch].contiguous() # collapse back to subgraph nodes

        # combine
        x = x_subgraph + x_global + x_clustered
        return x
    
class OwnSubgraphLayer(nn.Module):
    def __init__(self, emb_dim, conv_type='GIN', within_conv_dropout=0.):
        super().__init__()
        
        # subgraph message passing
        self.sg_conv = conv_getter(conv_type=conv_type, in_channels=emb_dim, out_channels=emb_dim, p_drop=within_conv_dropout)

        # within subgraph transformer
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=1,
            dropout=within_conv_dropout,
            batch_first=True,   # so input is [B, L, D]
        )
        self.mha_ln = nn.LayerNorm(emb_dim)

        # MLPs
        self.token_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, sg_x, sg_edge_attr, batch):

        # message passing within subgraph
        x_sg = self.sg_conv(sg_x, batch.sg_edge_index, sg_edge_attr)
        x_sg = self.node_mlp(x_sg)

        # attention over subgraph tokens
        # build cross-input for transformer
        sg_token = torch_scatter.scatter_mean(sg_x, batch.sg_batch, dim=0, dim_size=batch.num_subgraphs)  # [P_total, hidden]
        sg_token = self.mha_ln(sg_token)  # [P_total, hidden]
        tok_dense, mask = to_dense_batch(sg_token, batch.sg_graph)
        tok_transformed, _ = self.mha(tok_dense, tok_dense, tok_dense, key_padding_mask=~mask)  # [B, max_P_in_batch, hidden]
        sg_token = sg_token + tok_transformed[mask]  # [P_total, hidden]
        sg_token = sg_token + self.token_mlp(sg_token)
   
        # then add broadcasted token to each subgraph node
        x_sg = x_sg + sg_token[batch.sg_batch].contiguous()


        return x_sg
     

# =============================================================================
# 2. Subgraph Model Class
# =============================================================================

class SubgraphGNN(BaseModel):
    """Main explicit-subgraph GNN used for PSA.

    Args:
        cfg: Experiment configuration specifying the subgraph mode and model
            hyperparameters.
        data_stats: Dataset-derived feature, marking, and target metadata.

    Returns:
        A graph-level model that runs message passing on explicit subgraph-node
        copies, pools to subgraphs, then pools to whole-graph predictions.
    """
    def __init__(self, cfg, data_stats):
        super().__init__(cfg, data_stats)

        self.mode = cfg.model.mode
        self.within_conv_agg = cfg.model.get("within_conv_agg", "mean")
        self.cluster_agg = cfg.model.get("cluster_agg", "mean")
        assert self.cluster_agg in ["sum", "mean"] and self.within_conv_agg in ["sum", "mean"]
        
        self.gnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.intermediary_ffns = nn.ModuleList()

        for _ in range(self.num_layers):
            match self.mode:
                case 'psa':
                    sg_layer = PSALayer(
                        in_dim=self.embedding_dim,
                        out_dim=self.embedding_dim,
                        conv_type=self.conv_type,
                        within_conv_dropout=self.within_conv_dropout,
                        aggregation=self.within_conv_agg,
                    )
                case _:
                    raise ValueError(f"Unknown SubgraphGNN mode: {self.mode}")
                
            self.gnns.append(sg_layer)
            # use graph norm for BREC due to variance collapse for hard pairs of graphs
            if self.dataset_name == "BREC":
                self.norms.append(GraphNorm(self.embedding_dim))
            else:
                self.norms.append(nn.BatchNorm1d(self.embedding_dim))

            # interlayer MLP
            mlp = FFN_Layer(
                in_dim=self.embedding_dim,
                hidden_dim=self.embedding_dim,
                out_dim=self.embedding_dim,
                p_drop=self.head_dropout,
            ) if self.interlayer_mlp else None
            self.intermediary_ffns.append(mlp)

        # ---- global models at the end ----

        self.subgraph_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.head_dropout),
        ) if self.use_subgraph_mlp else None

        self.head = MLP_Head(
            in_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            out_dim=self.out_dim,
            p_drop=self.head_dropout,
        )
    
    def subgraph_pooler(self, x, batch):
        return global_add_pool(x, batch) if self.cluster_agg == "sum" else global_mean_pool(x, batch)

    def forward(self, batch):
        # 0) embed original node + edge features
        batch.x, batch.edge_attr = self.embed_node_edges(batch.x, batch.edge_attr)

        # 1) Build subgraph-node features
        x = self._build_subgraph_node_features(batch)  # [Tn_total, in_channels + mark_dim]

        # 2) subgraph edges + edge features
        edge_attr = self._build_subgraph_edge_attr(batch)  # [Te_total, total_edge_in_dim] or None

        # 3) Message passing over union-of-subgraphs graph
        for layer in range(self.num_layers):
            x_old = x

            # perform message passing in subgraph and aggregate representation from various subgraphs
            x = self.gnns[layer](x, edge_attr, batch)

            # batchnorm, activation, dropout
            x = self.norms[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.conv_dropout, training=self.training)
            x = x + x_old # residual connection

            # intermediary MLP
            if self.interlayer_mlp:
                x = self.intermediary_ffns[layer](x)


        # 4) pool nodes → subgraphs
        h_s = self.subgraph_pooler(x, batch.sg_batch)   # [P_total, hidden]

        # 5) subgraph-level MLP
        if self.use_subgraph_mlp:
            h_s = self.subgraph_mlp(h_s)               # [P_total, hidden]

        # 6) pool subgraphs → graphs
        h_g = self.graph_level_pooler(h_s, batch.sg_graph)  # [G, hidden]

        # 7) graph-level MLP head
        out = self.head(h_g)                       # [G, out_channels]
        return out
