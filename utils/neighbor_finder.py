import numpy as np


class NeighborFinder:
  def __init__(self, data, seed=None):
    """
    Build temporal adjacency lists from a Data-like object with fields:
      - sources, destinations, edge_idxs, timestamps (numpy arrays)
    """
    max_node_idx = int(max(data.sources.max(), data.destinations.max()))
    adj_list = [[] for _ in range(max_node_idx + 1)]

    for s, d, eidx, ts in zip(data.sources, data.destinations, data.edge_idxs, data.timestamps):
      adj_list[int(s)].append((int(d), int(eidx), float(ts)))
      adj_list[int(d)].append((int(s), int(eidx), float(ts)))

    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # neighbors: list of tuples (neighbor, edge_idx, timestamp)
      # sort by timestamp
      neighbors_sorted = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in neighbors_sorted], dtype=np.int32))
      self.node_to_edge_idxs.append(np.array([x[1] for x in neighbors_sorted], dtype=np.int32))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in neighbors_sorted], dtype=np.float32))

    self.random_state = np.random.RandomState(seed) if seed is not None else None

  def find_before(self, src_idx, cut_time):
    """
    Return all interactions strictly before cut_time for node src_idx, sorted by time.
    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
    return (
      self.node_to_neighbors[src_idx][:i],
      self.node_to_edge_idxs[src_idx][:i],
      self.node_to_edge_timestamps[src_idx][:i],
    )

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    For each (source_node, timestamp) return up to n_neighbors most recent neighbors
    with interaction time < timestamp. Left-pad with zeros if fewer than n_neighbors.
    """
    assert len(source_nodes) == len(timestamps)

    neighbors = np.zeros((len(source_nodes), n_neighbors), dtype=np.int32)
    edge_times = np.zeros((len(source_nodes), n_neighbors), dtype=np.float32)
    edge_idxs = np.zeros((len(source_nodes), n_neighbors), dtype=np.int32)

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      src_neigh, src_eidx, src_ets = self.find_before(int(source_node), float(timestamp))

      if len(src_neigh) > 0 and n_neighbors > 0:
        src_neigh = src_neigh[-n_neighbors:]
        src_eidx = src_eidx[-n_neighbors:]
        src_ets = src_ets[-n_neighbors:]

        k = len(src_neigh)
        neighbors[i, n_neighbors - k:] = src_neigh
        edge_times[i, n_neighbors - k:] = src_ets
        edge_idxs[i, n_neighbors - k:] = src_eidx

    return neighbors, edge_idxs, edge_times
