import numpy as np
import torch
from collections import defaultdict

from modules.layers import MergeLayer
from modules.memory.memory import Memory
from modules.memory.message_aggregator import LastMessageAggregator
from modules.memory.memory_updater import GRUMemoryUpdater
from modules.embedding_module2 import GraphAttentionEmbedding
from modules.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, memory_dimension=172, time_dimension=172,
               n_neighbors=None, aggregator_type="last",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               memory_updater_type="gru",
               n_nodes=9228):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device

    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_nodes = n_nodes
    #self.n_nodes = self.node_raw_features.shape[0] #???????????????????????????????? może -1? W preprocessingu dodajemy jeden pusty wiersz o indeksie 0
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = memory_dimension
    self.n_neighbors = n_neighbors
                 
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message

    self.time_encoder = TimeEncode(dimension=time_dimension)


    self.memory_dimension = memory_dimension
    message_dimension = 2 * self.memory_dimension + self.n_edge_features + self.time_encoder.dimension
                 
    self.memory = Memory(n_nodes=self.n_nodes, memory_dimension=self.memory_dimension, device=device)

    self.message_aggregator = LastMessageAggregator(device=device)
                 
    self.memory_updater = GRUMemoryUpdater(memory=self.memory,
                                           message_dimension=message_dimension,
                                           memory_dimension=self.memory_dimension,
                                           device=device)
 
    self.embedding_module = GraphAttentionEmbedding(edge_features=self.edge_raw_features,
                                                    neighbor_finder=self.neighbor_finder,
                                                    time_encoder=self.time_encoder,
                                                    n_layers=self.n_layers,
                                                    memory_dim=self.memory_dimension,
                                                    n_edge_features=self.n_edge_features,
                                                    n_time_features=time_dimension,
                                                    device=self.device,
                                                    n_heads=n_heads,
                                                    dropout=dropout)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.memory_dimension, self.memory_dimension, self.memory_dimension, 1)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = self.memory.get_memory(list(range(self.n_nodes)))
    last_update = self.memory.last_update

    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    unique_sources, source_id_to_messages = self.get_messages(source_nodes, source_node_embedding,
                                                              destination_nodes, destination_node_embedding,
                                                              edge_times, edge_idxs)
    unique_destinations, destination_id_to_messages = self.get_messages(destination_nodes, destination_node_embedding,
                                                                        source_nodes, source_node_embedding,
                                                                        edge_times, edge_idxs)
    self.update_memory(unique_sources, source_id_to_messages)
    self.update_memory(unique_destinations, destination_id_to_messages)

    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding, negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_messages(self, source_nodes, source_node_embedding, destination_nodes, destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features, source_time_delta_encoding], dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
