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
            use_pe: bool = True,
            use_pe_lin: bool = True,
            use_graph_feat_mlp: bool = True,
            use_edge_encoder: bool = True,
            use_node_feat_lin: bool = True,
    ):
        super().__init__()
        if use_pe:
            self.pe_dim = pe_dim
            self.pe_input_dim = pe_input_dim
        else:
            self.pe_dim = 0
            self.pe_input_dim = 0
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.graph_feat_dim = graph_feat_dim
        self.use_pe = use_pe
        self.use_pe_lin = use_pe_lin
        if not self.use_pe:
            self.use_pe_lin = False
        self.use_graph_feat_mlp = use_graph_feat_mlp
        self.use_edge_encoder = use_edge_encoder
        self.use_node_feat_lin = use_node_feat_lin
        self.channels = channels
        self.calculate_input_size()


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.use_pe_lin:
            self.node_feat_lin = nn.Linear(node_in_dim, channels - self.pe_dim)
        else:
            self.node_feat_lin = nn.Linear(node_in_dim, channels - self.pe_input_dim)

        if self.use_pe:
            self.pe_norm = nn.BatchNorm1d(self.pe_input_dim)
            if self.use_pe_lin:
                self.pe_lin = nn.Linear(self.pe_input_dim, self.pe_dim)
            else:
                self.pe_lin = None
                
        self.graph_feat_mlp = nn.Sequential(
            nn.Linear(graph_feat_dim, channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channels, channels)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, self.edge_out_dim),
            nn.ReLU(),
            nn.Linear(self.edge_out_dim, self.edge_out_dim),
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nn_gine = nn.Sequential(
                nn.Linear(self.padded_node_feature_size, self.padded_node_feature_size),
                nn.ReLU(),
                nn.Linear(self.padded_node_feature_size, self.padded_node_feature_size),
            )
            gine = GINEConv(nn=nn_gine, edge_dim=self.edge_feature_size)
            conv = GPSConv(
                channels=self.padded_node_feature_size,
                conv=gine,
                heads=4,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )
            self.convs.append(conv)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_mlp_input_size, self.edge_mlp_input_size // 2),
            nn.ReLU(),
            nn.Linear(self.edge_mlp_input_size // 2, 1),
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
    
    def calculate_input_size(self):
        """Calculates the effective input size for the convolutional layers."""
        if self.use_pe:
            if self.use_pe_lin:
                size_pe = self.pe_dim
            else:
                size_pe = self.pe_input_dim
        else:
            size_pe = 0
        
        if self.use_node_feat_lin:
            size_node = self.channels - size_pe
        else:
            size_node = self.node_in_dim
        
        if self.use_graph_feat_mlp: 
            size_graph = self.channels
        else:
            size_graph = self.graph_feat_dim
        
        self.node_feature_size = size_node + size_pe + size_graph
        # Calculate padded size to make it divisible by 4
        self.padded_node_feature_size = ((self.node_feature_size + 3) // 4) * 4
        self.padding_size = self.padded_node_feature_size - self.node_feature_size

        if not self.use_node_feat_lin:
            self.edge_out_dim = int(self.padded_node_feature_size / 2)
        else:
            self.edge_out_dim = self.channels

        if self.use_edge_encoder:
            size_edge = self.edge_out_dim
        else:
            size_edge = self.edge_in_dim
        
        self.edge_feature_size = size_edge
        self.edge_mlp_input_size = 3 * self.padded_node_feature_size + self.edge_feature_size

    def pad_features(self, x):
        """Pads the feature tensor to make it divisible by 4."""
        if self.padding_size > 0:
            padding = torch.zeros(x.size(0), self.padding_size, device=x.device)
            return torch.cat([x, padding], dim=-1)
        return x

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

        if self.use_node_feat_lin:
            x_emb = self.node_feat_lin(x.float())
        else:
            x_emb = x

        if self.use_pe:
            pe = self.pe_norm(pe)
            # Use positional encoding linear layer if specified
            if self.use_pe_lin:
                pe = self.pe_lin(pe)
            x = torch.cat([x_emb, pe], dim=-1)
        else:
            x = x_emb

        graph_feat = graph_feat.to(self.device)
        # Use graph feature MLP if specified
        if self.use_graph_feat_mlp:
            gf_emb = self.graph_feat_mlp(graph_feat)
        else:
            gf_emb = graph_feat

        if batch is not None:
            x = torch.cat([x, gf_emb[batch]], dim=-1)
        else:
            x = torch.cat([x, gf_emb], dim=-1)

        # Add padding if needed
        x = self.pad_features(x)

        if self.use_edge_encoder:
            edge_attr_enc = self.edge_encoder(edge_attr.float())
        else:
            edge_attr_enc = edge_attr

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
