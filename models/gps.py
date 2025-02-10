# models/gps.py
"""
GPS Model Module

Defines the GPS model for graph transformer tasks and related utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool
from torch_geometric.nn.attention import PerformerAttention


class RedrawProjection:
    """
    Handles redrawing of projection matrices in PerformerAttention layers.

    Args:
        model (nn.Module): The model containing attention layers.
        redraw_interval (Optional[int]): Interval for redrawing projections.
    """

    def __init__(self, model: nn.Module, redraw_interval: int = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        """Redraws projection matrices if the redraw interval is reached."""
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            for module in self.model.modules():
                if isinstance(module, PerformerAttention):
                    module.redraw_projection_matrix()
            self.num_last_redraw = 0
        else:
            self.num_last_redraw += 1


class GPS(nn.Module):
    """
    Graph Transformer model using GPSConv and GINEConv for edge-level tasks.

    Args:
        channels (int): Hidden dimension.
        pe_dim (int): Positional encoding dimension.
        num_layers (int): Number of layers.
        edge_in_dim (int): Dimension of edge input features.
        attn_type (str): Type of attention.
        attn_kwargs (dict): Additional attention parameters.
        pe_input_dim (int): Input dimension for positional encoding.
        graph_feat_dim (int): Graph-level feature dimension.
        node_in_dim (int): Dimension of node input features.
    """

    def __init__(
            self,
            channels: int,
            pe_dim: int,
            num_layers: int,
            edge_in_dim: int,
            attn_type: str,
            attn_kwargs: dict,
            pe_input_dim: int = 20,
            graph_feat_dim: int = 4,
            node_in_dim: int = 1,
    ):
        super().__init__()
        self.pe_dim = pe_dim
        self.pe_input_dim = pe_input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_feat_lin = nn.Linear(node_in_dim, channels - pe_dim)
        self.pe_norm = nn.BatchNorm1d(self.pe_input_dim)
        self.pe_lin = nn.Linear(self.pe_input_dim, self.pe_dim)
        self.graph_feat_mlp = nn.Sequential(
            nn.Linear(graph_feat_dim, channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channels, channels)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nn_gine = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
            )
            gine = GINEConv(nn=nn_gine, edge_dim=channels)
            conv = GPSConv(
                channels=channels,
                conv=gine,
                heads=4,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )
            self.convs.append(conv)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None
        )

    def extract_graph_features(self, data):
        """Extracts graph-level features from a Data object."""
        if hasattr(data, "graph_feat"):
            graph_feat = data.graph_feat
        elif hasattr(data, "graph_attr"):
            graph_attr = data.graph_attr
            graph_feat = torch.stack(
                [torch.tensor(value, dtype=torch.float32) for key, value in graph_attr.items() if key != "graph_id"],
                dim=0
            ).T
        else:
            raise AttributeError("No graph features found in Data object!")
        return graph_feat

    def forward(self, data):
        """
        Forward pass of the GPS model.

        Args:
            data: PyG Data object.

        Returns:
            Tensor of output logits for each edge.
        """
        x, pe = data.x, data.pe
        edge_index, edge_attr = data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        graph_feat = self.extract_graph_features(data)

        x_emb = self.node_feat_lin(x.float())
        pe = self.pe_norm(pe)
        pe = self.pe_lin(pe)
        x = torch.cat([x_emb, pe], dim=-1) if self.pe_dim > 0 else x_emb

        graph_feat = graph_feat.to(self.device)
        gf_emb = self.graph_feat_mlp(graph_feat)
        if batch is not None:
            x = x + gf_emb[batch]
        else:
            x = x + gf_emb[0]

        edge_attr_enc = self.edge_encoder(edge_attr.float())
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr_enc)

        if batch is None:
            g_context = x.mean(dim=0, keepdim=True)
            row, col = edge_index
            edge_graph_context = g_context.expand(len(row), -1)
        else:
            g_context = global_mean_pool(x, batch)
            row, col = edge_index
            edge_graph_context = g_context[batch[row]]

        e = torch.cat([x[row], x[col], edge_attr_enc, edge_graph_context], dim=-1)
        out = self.edge_mlp(e).squeeze(-1)
        return out
