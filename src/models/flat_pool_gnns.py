from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
import torch_scatter
from .layers import MLP_Head, conv_getter, BaseModel, FFN_Layer

import torch
from torch_geometric.utils import coalesce, remove_self_loops

from torch_geometric.nn.norm import GraphNorm

class SubgraphPoolingGNN(BaseModel):
    """Pool original nodes into configured subgraphs, then run graph prediction.

    Args:
        cfg: Experiment configuration specifying the pre-pooling, within-cluster,
            and post-pooling message-passing layout.
        data_stats: Dataset-derived feature and target metadata.

    Returns:
        A graph-level model that optionally performs message passing before and
        within clusters, pools to subgraphs, and predicts from the pooled graph.
    """
    def __init__(self, cfg, data_stats):
        super().__init__(cfg, data_stats)

        # Settings & Argumnets
        self.num_initial_mp = cfg.model.get("number_initial_mp", 0)
        self.num_post_mp = cfg.model.get("number_post_mp", 0)

        self.within_cluster_mp = cfg.model.get("within_cluster_mp", False)
        self.num_within_mp = cfg.model.get("number_within_mp", 0)

        assert self.within_cluster_mp if self.num_within_mp > 0 else True, "num_within_mp > 0 requires within_cluster_mp=True"
        assert self.num_within_mp > 0 if self.within_cluster_mp else True, "num_within_mp must be positive"
        print(f"Using within-cluster MP: {self.within_cluster_mp} with {self.num_within_mp} layers.")
        
        self.within_conv_agg = cfg.model.get("within_conv_agg", "mean")
        self.cluster_agg = cfg.model.get("cluster_agg", "mean")
        assert self.cluster_agg in ["sum", "mean"] and self.within_conv_agg in ["sum", "mean"]
        
        self.conv_agg_func = torch_scatter.scatter_mean if self.within_conv_agg == "mean" else torch_scatter.scatter_add   
        self.cluster_agg_func = torch_scatter.scatter_mean if self.cluster_agg == "mean" else torch_scatter.scatter_add   


        # define global stack
        self.initial_convs, self.initial_bns, self.initial_mlps = self._instantiate_mp_stack(self.num_initial_mp, self.conv_type)

        # define option within-cluster stack
        if self.within_cluster_mp:
            self.within_convs, self.within_bns, self.within_mlps = self._instantiate_mp_stack(self.num_within_mp, "GIN")
        # define clustered stack
        self.post_convs, self.post_bns, self.post_mlps = self._instantiate_mp_stack(self.num_post_mp, "GIN") 
        # for post-pooling, we only use GIN since we don't have edge features anymore

        # define subgraph pooler & head
        self.head = MLP_Head(
            in_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            out_dim=self.out_dim,
            p_drop=self.head_dropout,
        )

    def _instantiate_mp_stack(self, num_layers, conv_type):
        stack_convs = nn.ModuleList()
        stack_bns = nn.ModuleList()
        stack_mlps = nn.ModuleList()

        for _ in range(num_layers):
            conv = conv_getter(
                conv_type,
                self.embedding_dim,
                self.embedding_dim,
                use_norm=(self.dataset_name!="BREC"),  # use batchnorm in GIN except for BREC 
            )
            stack_convs.append(conv)
            stack_bns.append(GraphNorm(self.embedding_dim) 
                             if self.dataset_name=="BREC" else 
                             nn.BatchNorm1d(self.embedding_dim)
                             )

            mlp = FFN_Layer(
                in_dim=self.embedding_dim,
                hidden_dim=self.embedding_dim,
                out_dim=self.embedding_dim,
                p_drop=self.head_dropout,
            ) if self.interlayer_mlp else None
            stack_mlps.append(mlp)

        return stack_convs, stack_bns, stack_mlps
    

    def _perform_mp_stack(self, x, edge_index, edge_attr, stack_convs, stack_bns, stack_mlps):
        for layer in range(len(stack_convs)):
            x_old = x  # update for next residual
            
            # conv
            if self.conv_type == 'GINE' and edge_attr is not None:
                x = stack_convs[layer](x, edge_index, edge_attr)
            else:
                x = stack_convs[layer](x, edge_index)

            x = stack_bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.conv_dropout, training=self.training)
            x = x + x_old  # update for next residual

            if stack_mlps[layer] is not None:
                x = stack_mlps[layer](x)

        return x
    
    def subgraph_pooler(self, x, batch):
        return self.pool(x, batch) # NOTE: uses same function as graph-level pooler

    def forward(self, batch):
        # embed raw features
        batch.x, batch.edge_attr = self.embed_node_edges(batch.x, batch.edge_attr)
        x, edge_attr = batch.x, batch.edge_attr
        edge_index = batch.edge_index

        # ---- initial global MP ----
        x = self._perform_mp_stack(x, edge_index, edge_attr, self.initial_convs, self.initial_bns, self.initial_mlps)

        # optional within-cluster MP before pooling
        if self.within_cluster_mp:
            # do conv in subgraph space
            x_sg = x[batch.sg_nodes_orig] 
            edge_index_sg = batch.sg_edge_index
            # actually x_sg should be equal to x, since here we wish to have a disjoint clustering
            # assert x_sg.size(0) == x.size(0), "For disjoint clustering, all nodes must belong to some unique cluster."
            x_sg = self._perform_mp_stack(x_sg, edge_index_sg, None, self.within_convs, self.within_bns, self.within_mlps)
            
            # x[batch.sg_nodes_orig] = x_sg  # update back to original node space
            x = self.conv_agg_func(
                x_sg,
                batch.sg_nodes_orig,
                dim=0,
                dim_size=batch.num_nodes,
            )  # [N_total, hidden]
        
        # ---- pool to subgraph space ----
        # x: [N, F]
        # expand features to membership entries and aggregate for each cluster
        cluster_x = self.cluster_agg_func(x[batch.sg_nodes_orig], batch.sg_batch, dim=0, dim_size=batch.num_subgraphs)

        # ---- hierarchical MP ----
        cluster_x = self._perform_mp_stack(cluster_x, batch.super_edge_index, None,
                                          self.post_convs, self.post_bns, self.post_mlps)

        # ---- readout ----
        h_g = self.graph_level_pooler(cluster_x, batch.sg_graph)  # subgraphs -> graphs
        return self.head(h_g)


@torch.no_grad()
def _sanity_check_partition(batch: torch.Tensor, pool_assign: torch.Tensor):
    """
    Optional debug check: each supernode should belong to exactly one graph.
    """
    if batch is None:
        return
    # each cluster should map to a single batch id
    bmax, _ = scatter_max(batch, pool_assign, dim=0)
    bmin = -scatter_max(-batch, pool_assign, dim=0)[0]  # scatter_min without importing
    if not torch.equal(bmax, bmin):
        raise ValueError(
            "pool_assign is not a per-graph partition after batching: "
            "a supernode id appears in multiple graphs. "
            "This usually means pool_assign was not offset correctly."
        )


def hard_pool_from_assign(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pool_assign: torch.Tensor,
    batch: torch.Tensor | None = None,
    num_supernodes_total: int | None = None,
    return_edge_weight: bool = True,
    remove_self_loops_super: bool = True,
    sanity_check: bool = False,
):
    """
    Hard pooling operator for (batched) graphs using a supernode assignment.

    Args:
      x: [N, F]
      edge_index: [2, E]
      pool_assign: [N]  supernode id for each node (must be globally offset if batched)
      batch: [N]  graph id per node (optional; inferred as all zeros if None)
      num_supernodes_total: total number of supernodes (optional)
      return_edge_weight: if True, returns multiplicity of contracted edges
      remove_self_loops_super: remove self-loops on super graph
      sanity_check: if True, verify partition is valid across batch

    Returns:
      x_super: [P, F]
      edge_index_super: [2, Es]
      edge_weight_super: [Es]  (or None if return_edge_weight=False)
      super_batch: [P]
    """
    device = x.device
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    if num_supernodes_total is None:
        num_supernodes_total = int(pool_assign.max().item() + 1) if pool_assign.numel() > 0 else 0

    if sanity_check:
        _sanity_check_partition(batch, pool_assign)

    # ---- 1) Sum-collapse features (injective aggregation) ----
    x_super = scatter_add(x, pool_assign, dim=0, dim_size=num_supernodes_total)

    # ---- 2) super_batch: which graph each supernode belongs to ----
    # For a valid per-graph partition, all nodes in a supernode share same batch id.
    super_batch, _ = scatter_max(batch, pool_assign, dim=0, dim_size=num_supernodes_total)

    # ---- 3) Contract edges ----
    if edge_index.numel() == 0 or num_supernodes_total == 0:
        edge_index_super = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight_super = torch.empty((0,), dtype=torch.float, device=device) if return_edge_weight else None
        return x_super, edge_index_super, edge_weight_super, super_batch

    u, v = edge_index[0], edge_index[1]
    su = pool_assign[u]
    sv = pool_assign[v]

    edge_index_super = torch.stack([su, sv], dim=0)

    if return_edge_weight:
        # each original edge contributes weight 1
        edge_weight = torch.ones(edge_index_super.size(1), device=device, dtype=torch.float)
        edge_index_super, edge_weight_super = coalesce(
            edge_index_super, edge_weight, num_nodes=num_supernodes_total
        )
    else:
        edge_index_super = coalesce(edge_index_super, None, num_nodes=num_supernodes_total)[0]
        edge_weight_super = None

    if remove_self_loops_super:
        edge_index_super, edge_weight_super = remove_self_loops(edge_index_super, edge_weight_super)

    return x_super, edge_index_super, edge_weight_super, super_batch
