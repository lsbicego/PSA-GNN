from torch import nn
import torch.nn.functional as F
from .layers import MLP_Head, conv_getter, BaseModel, FFN_Layer

class MP_GNN(BaseModel):
    """Standard graph-level message passing baseline.

    Args:
        cfg: Experiment configuration with architecture and optimization
            settings.
        data_stats: Dataset-derived feature and target metadata.

    Returns:
        A model that performs message passing on the original graph and then
        applies graph-level pooling and an MLP head.
    """

    def __init__(self, cfg, data_stats):
        super().__init__(cfg, data_stats)

        # layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.intermediary_ffns = nn.ModuleList()
        self.gps_attn_type = cfg.model.get("gps_attn_type", "performer")

        for _ in range(self.num_layers):
            conv = conv_getter(
                self.conv_type,
                self.embedding_dim,
                self.embedding_dim,
                gps_attn_type=self.gps_attn_type,
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(self.embedding_dim))

            mlp = FFN_Layer(
                in_dim=self.embedding_dim,
                hidden_dim=self.embedding_dim,
                out_dim=self.embedding_dim,
                p_drop=self.head_dropout,
            ) if self.interlayer_mlp else None
            self.intermediary_ffns.append(mlp)

        self.head = MLP_Head(
            in_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            out_dim=self.out_dim,
            p_drop=self.head_dropout,
        )

    def forward(self, batch):
        # embed raw node/edge features
        batch.x, batch.edge_attr = self.embed_node_edges(batch.x, batch.edge_attr)
        x, edge_attr = batch.x, batch.edge_attr
        edge_index = batch.edge_index

        for layer in range(self.num_layers):
            # convolution
            if self.conv_type == 'GINE' and edge_attr is not None:
                x_new = self.convs[layer](x, edge_index, edge_attr)
            else:
                x_new = self.convs[layer](x, edge_index)

            # batch norm, activation, dropout, residual
            x_new = self.bns[layer](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.conv_dropout, training=self.training)
            x = x + x_new  # residual connection
            
            # optional inter-layer MLP with skip connection
            if self.intermediary_ffns[layer] is not None:
                x = self.intermediary_ffns[layer](x) # NOTE: includes BN(x + MLP(x)) internally

        # graph-level pooling and head
        h_g = self.graph_level_pooler(x, batch.batch)
        return self.head(h_g)
