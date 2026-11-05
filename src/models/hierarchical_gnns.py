import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense import (
    DenseGCNConv,
    DenseGINConv,
    dense_diff_pool,
    dense_mincut_pool,
)
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.norm import GraphNorm

from torch_geometric.nn import (
    TopKPooling,
    ASAPooling,
    SAGPooling,
    EdgePooling,
)

from .layers import BaseModel, MLP_Head, FFN_Layer

class DenseAssignNet(nn.Module):
    """Predict dense DiffPool assignments from batched dense graphs.

    Args:
        in_dim: Input node embedding size.
        out_clusters: Number of pooled clusters to predict.
        conv_type: Dense convolution family used in the assignment network.
        conv_dropout: Dropout applied between assignment layers.
        num_layers: Number of assignment layers.

    Returns:
        A module that maps dense node features to assignment scores.
    """
    def __init__(self, in_dim, out_clusters, conv_type, conv_dropout=0.0, num_layers=1):
        super().__init__()
        assert num_layers in [1, 2], "assign_layers must be 1 or 2."

        self.convs = nn.ModuleList()
        self.conv_type = conv_type
        self.conv_dropout = conv_dropout
        self.num_layers = num_layers

        if conv_type == "GCN":
            # first (or only) layer
            if num_layers == 1:
                self.convs.append(DenseGCNConv(in_dim, out_clusters))
            else:
                self.convs.append(DenseGCNConv(in_dim, in_dim))
                self.convs.append(DenseGCNConv(in_dim, out_clusters))

        elif conv_type == "GIN" or conv_type=="GINE":
            def make_mlp(in_ch, out_ch):
                return nn.Sequential(
                    nn.Linear(in_ch, in_ch),
                    nn.ReLU(),
                    nn.Linear(in_ch, out_ch),
                )

            if num_layers == 1:
                self.convs.append(DenseGINConv(make_mlp(in_dim, out_clusters)))
            else:
                self.convs.append(DenseGINConv(make_mlp(in_dim, in_dim)))
                self.convs.append(DenseGINConv(make_mlp(in_dim, out_clusters)))
        else:
            raise ValueError(
                f"DenseAssignNet supports conv_type in {{'GCN','GIN'}}, got {conv_type}"
            )

    def forward(self, x_dense, adj, mask=None):
        """Compute assignment scores for a dense batched graph.

        Args:
            x_dense: Dense node features of shape ``[B, N, F]``.
            adj: Dense adjacency tensor of shape ``[B, N, N]``.
            mask: Optional node mask of shape ``[B, N]``.

        Returns:
            Assignment scores of shape ``[B, N, out_clusters]``.
        """
        for li, conv in enumerate(self.convs):
            x_dense = conv(x_dense, adj, mask=mask)
            # Apply the nonlinearity only between assignment layers.
            if li < len(self.convs) - 1:
                x_dense = F.relu(x_dense)
                x_dense = F.dropout(
                    x_dense,
                    p=self.conv_dropout,
                    training=self.training,
                )
        return x_dense  # used directly as S (softmax inside dense_diff_pool)

class HierarchicalSparsePoolingGNN(BaseModel):
    """Hierarchical model with sparse pooling between message-passing stages.

    Args:
        cfg: Experiment configuration specifying pooling type, stage counts, and
            message-passing hyperparameters.
        data_stats: Dataset-derived feature and target metadata.

    Returns:
        A graph-level model that alternates sparse pooling with message passing
        over progressively coarsened graphs.
    """
    def __init__(self, cfg, data_stats):
        super().__init__(cfg, data_stats)

        # pooling config
        self.pooling_type = cfg.model.pooling_type.lower()
        self.pool_ratio = getattr(cfg.model, "pool_ratio", 0.5)

        # hierarchical layout configuration
        self.num_initial_mp = int(getattr(cfg.model, "number_initial_mp", 1))
        self.num_between_mp = int(getattr(cfg.model, "number_between_mp", 1))
        self.num_post_mp = int(getattr(cfg.model, "number_post_mp", 1))
        default_pool_steps = max(self.num_layers - 2, 0)
        self.num_pooling_steps = int(
            getattr(cfg.model, "number_pooling_steps", default_pool_steps)
        )

        if self.num_initial_mp < 0 or self.num_between_mp < 0 or self.num_post_mp < 0:
            raise ValueError("Message passing counts must be non-negative.")
        if self.num_pooling_steps < 0:
            raise ValueError("number_pooling_steps must be non-negative.")

        # ---- build conv + pool stacks ----
        self.initial_stage = self._build_sparse_stage(self.num_initial_mp)
        self.between_stages = nn.ModuleList(
            [self._build_sparse_stage(self.num_between_mp) for _ in range(self.num_pooling_steps)]
        )
        self.post_stage = self._build_sparse_stage(self.num_post_mp)
        self.pools = nn.ModuleList(
            [self._make_sparse_pool() for _ in range(self.num_pooling_steps)]
        )

        # final graph-level head (same as in KernelSubgraphGNN)
        self.head = MLP_Head(
            in_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            out_dim=self.out_dim,
            p_drop=self.head_dropout,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_sparse_pool(self):
        """Instantiate a sparse pooling module."""
        if self.pooling_type == "topk":
            return TopKPooling(self.embedding_dim, ratio=self.pool_ratio)
        elif self.pooling_type == "asap":
            return ASAPooling(self.embedding_dim, ratio=self.pool_ratio)
        elif self.pooling_type == "sag":
            return SAGPooling(self.embedding_dim, ratio=self.pool_ratio)
        elif self.pooling_type == "edge_pool":
            return EdgePooling(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported sparse pooling_type: {self.pooling_type}")

    def _build_sparse_stage(self, num_layers):
        """Create a stack of sparse conv blocks."""
        stage = nn.ModuleDict(
            {
                "convs": nn.ModuleList(),
                "norms": nn.ModuleList(),
                "mlps": nn.ModuleList(),
            }
        )

        for _ in range(num_layers):
            conv = self.create_layer_conv()
            stage["convs"].append(conv)            
            # use graph norm for BREC due to variance collapse for hard pairs of graphs
            stage["norms"].append(nn.BatchNorm1d(self.embedding_dim) if 
                                  self.dataset_name!="BREC" else 
                                  GraphNorm(self.embedding_dim))  
            stage["mlps"].append(self._build_inter_mlp())
        return stage

    def _build_inter_mlp(self):
        if not self.interlayer_mlp:
            return None

        return FFN_Layer(
            in_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            out_dim=self.embedding_dim,
            p_drop=self.head_dropout,
        )

    def _run_sparse_stage(self, stage, x, edge_index, edge_attr):
        """Apply every block in a sparse stage sequentially."""
        x_old = x
        for conv, norm, mlp in zip(
            stage["convs"], stage["norms"], stage["mlps"]
        ):
            x, edge_attr = self.perform_conv(
                conv=conv,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.conv_dropout, training=self.training)
            x = x + x_old  # residual connection

            if self.interlayer_mlp and mlp is not None:
                x = mlp(x)
            
            x_old = x  # update for next residual

        return x, edge_attr

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch):
        # embed node & edge attributes (BaseModel logic)
        batch.x, batch.edge_attr = self.embed_node_edges(batch.x, batch.edge_attr)

        x = batch.x                      # [N, F]
        edge_index = batch.edge_index    # [2, E]
        edge_attr = batch.edge_attr      # [E, Fe] or None
        graph_batch = batch.batch        # [N]

        # 1) initial message passing stack
        x, edge_attr = self._run_sparse_stage(
            stage=self.initial_stage,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # 2) alternating pooling + between-stage message passing
        for pool, between_stage in zip(self.pools, self.between_stages):
            if self.pooling_type == "edge_pool":
                x, edge_index, graph_batch, _ = pool(x, edge_index, graph_batch)
            elif self.pooling_type == "asap":
                # Do not pass edge features to ASAPooling (expects scalar edge_weight).
                out = pool(x, edge_index, None, graph_batch)
                x, edge_index, _, graph_batch = out[0], out[1], out[2], out[3]
                if self.conv_type == "GINE":
                    # Keep GINE interface valid after pooling while ignoring edge features.
                    edge_attr = x.new_zeros((edge_index.size(1), self.embedding_dim))
                else:
                    edge_attr = None
            else:
                out = pool(x, edge_index, edge_attr, graph_batch)
                x, edge_index, edge_attr, graph_batch = out[0], out[1], out[2], out[3]

            x, edge_attr = self._run_sparse_stage(
                stage=between_stage,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        # 3) post-pooling message passing stack
        x, edge_attr = self._run_sparse_stage(
            stage=self.post_stage,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # 4) global pooling to graph level
        h_g = self.graph_level_pooler(x, graph_batch)  # uses BaseModel.pool

        # 5) MLP head
        out = self.head(h_g)
        return out


class HierarchicalDensePoolingGNN(BaseModel):
    """Hierarchical model with dense assignment-based pooling.

    Args:
        cfg: Experiment configuration specifying dense pooling type and stage
            hyperparameters.
        data_stats: Dataset-derived feature and target metadata.

    Returns:
        A graph-level model that applies dense pooling layers such as DiffPool
        or MinCutPool between message-passing stages.
    """
    def __init__(self, cfg, data_stats):
        super().__init__(cfg, data_stats)

        assert self.num_layers >= 2, \
            "num_layers must be at least 2 (initial + final conv)."
        default_pool_steps = max(self.num_layers - 2, 0)
        self.num_initial_mp = int(getattr(cfg.model, "number_initial_mp", 1))
        self.num_between_mp = int(getattr(cfg.model, "number_between_mp", 1))
        self.num_post_mp = int(getattr(cfg.model, "number_post_mp", 1))
        self.num_pooling_steps = int(
            getattr(cfg.model, "number_pooling_steps", default_pool_steps)
        )

        if self.num_initial_mp < 0 or self.num_between_mp < 0 or self.num_post_mp < 0:
            raise ValueError("Message passing counts must be non-negative.")
        if self.num_pooling_steps < 0:
            raise ValueError("number_pooling_steps must be non-negative.")

        # which dense pooling scheme
        self.pooling_type = cfg.model.pooling_type.lower()  # 'diffpool' or 'mincut'

        if self.pooling_type not in {"diffpool", "mincut"}:
            raise ValueError(f"Unsupported dense pooling_type: {self.pooling_type}")

        # cluster configuration for each pooling layer
        pool_clusters = getattr(cfg.model, "pool_ratio", None)
        if pool_clusters is None:
            raise ValueError(
                "cfg.model.pool_ratio must be set for dense pooling "
                "(int, float, or list[int], one per pooling layer)."
            )

        if isinstance(pool_clusters, int):
            # fixed number of clusters per layer
            self.pool_clusters = [pool_clusters] * self.num_pooling_steps
        elif isinstance(pool_clusters, float):
            # ratio * max_nodes -> integer clusters
            self.pool_clusters = [
                max(2, int(round(data_stats['max_nodes'] * pool_clusters)))
            ] * self.num_pooling_steps
        else:
            # assume iterable / list
            self.pool_clusters = list(pool_clusters)
            if len(self.pool_clusters) != self.num_pooling_steps:
                raise ValueError(
                    f"len(pool_clusters)={len(self.pool_clusters)} "
                    f"does not match number_pooling_steps={self.num_pooling_steps}"
                )

        # assignment GNN depth for DiffPool: 1 or 2 layers
        self.assign_layers = getattr(cfg.model, "assign_layers", 1)

        # ---- sparse input conv stack ----
        self.initial_sparse_stage = self._build_sparse_stage(self.num_initial_mp)

        # ---- dense convs + assignment heads for pooling layers ----
        self.between_dense_stages = nn.ModuleList(
            [self._build_dense_stage(self.num_between_mp) for _ in range(self.num_pooling_steps)]
        )

        # For DiffPool: assignment via DenseAssignNet
        # For MinCut:  assignment via simple linear projection
        if self.pooling_type == "diffpool":
            self.assign_gnns = nn.ModuleList()
            self.assign_mlps = None
        else:  # "mincut"
            self.assign_gnns = None
            self.assign_mlps = nn.ModuleList()

        for i in range(self.num_pooling_steps):
            # assignment branch
            if self.pooling_type == "diffpool":
                self.assign_gnns.append(
                    DenseAssignNet(
                        in_dim=self.embedding_dim,
                        out_clusters=self.pool_clusters[i],
                        conv_type=self.conv_type,
                        conv_dropout=self.conv_dropout,
                        num_layers=self.assign_layers,
                    )
                )
            else:  # "mincut"
                self.assign_mlps.append(
                    nn.Linear(self.embedding_dim, self.pool_clusters[i])
                )

        # ---- final dense conv stack (no pooling) ----
        self.post_dense_stage = self._build_dense_stage(self.num_post_mp)

        # ---- graph-level head ----
        self.head = MLP_Head(
            in_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            out_dim=self.out_dim,
            p_drop=self.head_dropout,
        )

        # will store pooling regularization losses
        self.reg_loss = torch.tensor(0.0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_dense_conv(self):
        """Create a DenseConv matching self.conv_type."""
        if self.conv_type == "GCN":
            return DenseGCNConv(self.embedding_dim, self.embedding_dim)
        elif self.conv_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Dropout(self.conv_dropout),
            )
            return DenseGINConv(mlp)
        elif self.conv_type == "GINE":
            mlp = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Dropout(self.conv_dropout),
            )
            print("Using GIN instead of GINE in dense conv block.")
            return DenseGINConv(mlp)
        else:
            raise ValueError(
                f"Dense pooling currently supports conv_type in {{'GCN','GIN'}}, "
                f"got {self.conv_type}"
            )

    def _build_inter_mlp(self):
        if not self.interlayer_mlp:
            return None
        return FFN_Layer(
            in_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            out_dim=self.embedding_dim,
            p_drop=self.head_dropout,
        )

    def _build_sparse_stage(self, num_layers):
        stage = nn.ModuleDict(
            {
                "convs": nn.ModuleList(),
                "norms": nn.ModuleList(),
                "mlps": nn.ModuleList(),
            }
        )

        for _ in range(num_layers):
            conv = self.create_layer_conv()
            stage["convs"].append(conv)
            # use graph norm for BREC due to variance collapse for hard pairs of graphs
            stage["norms"].append(nn.BatchNorm1d(self.embedding_dim) if 
                                  self.dataset_name!="BREC" else 
                                  GraphNorm(self.embedding_dim))
            stage["mlps"].append(self._build_inter_mlp())

        return stage

    def _build_dense_stage(self, num_layers):
        stage = nn.ModuleDict(
            {
                "convs": nn.ModuleList(),
                "norms": nn.ModuleList(),
                "mlps": nn.ModuleList(),
            }
        )

        for _ in range(num_layers):
            stage["convs"].append(self._make_dense_conv())
            # use graph norm for BREC due to variance collapse for hard pairs of graphs
            stage["norms"].append(nn.BatchNorm1d(self.embedding_dim) if 
                                  self.dataset_name!="BREC" else 
                                  GraphNorm(self.embedding_dim))  
            stage["mlps"].append(self._build_inter_mlp())

        return stage

    def _apply_dense_block(self, x_dense, adj, mask, conv, norm, inter_mlp):
        """
        Apply a dense conv + normalization + optional MLP + ReLU + dropout.

        x_dense: [B, N, F]
        adj:     [B, N, N]
        mask:    [B, N] or None
        """
        x_original = x_dense
        # DenseConv expects [B, N, F], [B, N, N], and optional mask
        x_dense = conv(x_dense, adj, mask=mask)  # [B, N, F]

        # convert to sparse again
        B, N, Fdim = x_dense.size()
        x_flat = x_dense.view(B * N, Fdim)

        x_flat = norm(x_flat)
        x_flat = F.relu(x_flat)
        x_flat = F.dropout(x_flat, p=self.conv_dropout, training=self.training)
        x_flat = x_flat + x_original.view(B * N, Fdim)  # residual connection

        if self.interlayer_mlp and inter_mlp is not None:
            x_flat = inter_mlp(x_flat)

        # convert back to dense
        x_dense = x_flat.view(B, N, Fdim)
        return x_dense

    def _apply_sparse_stage(self, stage, x, edge_index, edge_attr):
        for conv, norm, mlp in zip(
            stage["convs"], stage["norms"], stage["mlps"]
        ):
            x, edge_attr = self.perform_conv(
                conv=conv,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.conv_dropout, training=self.training)

            if self.interlayer_mlp and mlp is not None:
                x = mlp(x)
        return x, edge_attr

    def _apply_dense_stage(self, stage, x_dense, adj, mask):
        for conv, norm, inter_mlp in zip(
            stage["convs"], stage["norms"], stage["mlps"]
        ):
            x_dense = self._apply_dense_block(
                x_dense=x_dense,
                adj=adj,
                mask=mask,
                conv=conv,
                norm=norm,
                inter_mlp=inter_mlp,
            )
        return x_dense

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch):
        # embed node & edge attributes
        batch.x, batch.edge_attr = self.embed_node_edges(batch.x, batch.edge_attr)

        x = batch.x                      # [N, F]
        edge_index = batch.edge_index    # [2, E]
        edge_attr = batch.edge_attr      # [E, Fe] or None
        graph_batch = batch.batch        # [N]

        # ---- 0) sparse input conv stack ----
        x, edge_attr = self._apply_sparse_stage(
            stage=self.initial_sparse_stage,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # ---- 1) convert to dense ----
        # x_dense: [B, N_max, F], mask: [B, N_max]
        x_dense, mask = to_dense_batch(x, graph_batch)      # mask for valid nodes
        adj = to_dense_adj(edge_index, batch=graph_batch)   # [B, N_max, N_max]

        # ---- 2) hierarchical dense pooling ----
        self.reg_loss = x_dense.new_tensor(0.0)

        for i in range(self.num_pooling_steps):
            # assignment S_i before pooling
            if self.pooling_type == "diffpool":
                # via dense GNN
                S = self.assign_gnns[i](x_dense, adj, mask=mask)
                x_dense, adj, lp_loss, ent_loss = dense_diff_pool(
                    x_dense, adj, S, mask=mask
                )
                self.reg_loss = self.reg_loss + lp_loss + ent_loss
            else:  # 'mincut' -> linear projection
                S = self.assign_mlps[i](x_dense)
                x_dense, adj, mincut_loss, ortho_loss = dense_mincut_pool(
                    x_dense, adj, S, mask=mask
                )
                self.reg_loss = self.reg_loss + mincut_loss + ortho_loss

            # after pooling, all nodes are valid -> drop mask
            mask = None

            # dense conv blocks on pooled graph
            x_dense = self._apply_dense_stage(
                stage=self.between_dense_stages[i],
                x_dense=x_dense,
                adj=adj,
                mask=mask,
            )

        # ---- 3) final dense conv (no pooling) ----
        x_dense = self._apply_dense_stage(
            stage=self.post_dense_stage,
            x_dense=x_dense,
            adj=adj,
            mask=mask,
        )

        # ---- 4) global graph pooling ----
        B, N_final, Fdim = x_dense.size()
        x_flat = x_dense.view(B * N_final, Fdim)

        # construct a batch vector for flattened nodes
        graph_batch_dense = torch.arange(
            B, device=x_flat.device, dtype=torch.long
        ).repeat_interleave(N_final)

        h_g = self.graph_level_pooler(x_flat, graph_batch_dense)

        # ---- 5) MLP head ----
        out = self.head(h_g)
        return out
