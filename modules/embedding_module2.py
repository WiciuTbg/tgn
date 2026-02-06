import numpy as np
import torch
from torch import nn

from modules.temporal_attention import TemporalAttentionLayer


class GraphAttentionEmbedding(nn.Module):
    def __init__(
        self,
        edge_features,            # Fajnie jakbyśmy nie przekazywali od razu wszystkich edge_features
        time_encoder,
        n_neighbors: int,
        n_layers: int,
        memory_dim: int,
        n_edge_features: int,
        n_time_features: int,
        device,
        n_heads: int = 2,
        dropout: float = 0.1,
        neighbor_finder = None,
    ):
        super().__init__()
        self.edge_features = edge_features
        self.neighbor_finder = neighbor_finder
        self.n_neighbors = n_neighbors
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.device = device

        # One attention block per layer (same as the original design)
        self.attention_models = nn.ModuleList(
            [
                TemporalAttentionLayer(
                    n_node_features=memory_dim,
                    n_neighbors_features=memory_dim,
                    n_edge_features=n_edge_features,
                    time_dim=n_time_features,
                    n_head=n_heads,
                    dropout=dropout,
                    output_dimension=memory_dim,
                )
                for _ in range(n_layers)
            ]
        )

    def compute_embedding(
        self,
        memory,
        source_nodes,
        timestamps,
        n_layers = None,
    ):

        if n_layers is None:
          n_layers = self.n_layers
          
        """
        Recursive temporal graph attention.
        source_nodes: np.ndarray[int] shape (B,)
        timestamps:   np.ndarray[float] shape (B,)
        """
        assert n_layers >= 0

        src = torch.from_numpy(source_nodes).long().to(self.device)
        ts = torch.from_numpy(timestamps).float().to(self.device).unsqueeze(1)

        # Query node time span is always 0
        src_time_emb = self.time_encoder(torch.zeros_like(ts))
     
        if n_layers == 0:
            return memory[source_nodes, :]

        # Compute (n_layers-1)-hop embeddings for the source nodes
        src_conv = self.compute_embedding(
            memory=memory,
            source_nodes=source_nodes,
            timestamps=timestamps,
            n_layers=n_layers - 1,
        )

        neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes, timestamps, n_neighbors=self.n_neighbors)

        neighbors_t = torch.from_numpy(neighbors).long().to(self.device)
        edge_idxs_t = torch.from_numpy(edge_idxs).long().to(self.device)

        # Time deltas (query time - neighbor edge time)
        edge_deltas = timestamps[:, np.newaxis] - edge_times
        edge_deltas_t = torch.from_numpy(edge_deltas).float().to(self.device)

        # Flatten neighbors and recursively compute their embeddings
        flat_neighbors = neighbors.reshape(-1)
        neigh_emb = self.compute_embedding(
            memory=memory,
            source_nodes=flat_neighbors,
            timestamps=np.repeat(timestamps, self.n_neighbors),
            n_layers=n_layers - 1,
        )

        neigh_emb = neigh_emb.view(len(source_nodes), self.n_neighbors, -1)

        edge_time_emb = self.time_encoder(edge_deltas_t)
        edge_feat = self.edge_features[edge_idxs_t, :]

        # Mask padded neighbors (id == 0)
        mask = neighbors_t == 0

        # Apply the attention layer corresponding to this depth
        attn = self.attention_models[n_layers - 1]
        out, _ = attn(src_conv, src_time_emb, neigh_emb, edge_time_emb, edge_feat, mask)
        return out
