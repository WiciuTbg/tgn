import numpy as np
import random
import pandas as pd


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name, new_nodes_mode="original implementation"):
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  # Time masks
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  if new_nodes_mode == "original implementation":
    random.seed(2020)

    node_set = full_data.unique_nodes
    n_total_unique_nodes = full_data.n_unique_nodes

    # Nodes that appear after val_time (i.e., in val+test)
    post_val_mask = timestamps > val_time
    test_node_set = set(sources[post_val_mask]) | set(destinations[post_val_mask])

    new_test_node_set = set(
      random.sample(list(test_node_set), int(0.1 * n_total_unique_nodes))
    )

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    train_node_set = set(train_data.sources) | set(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)]
    )
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  elif new_nodes_mode == "real":
    train_mask = timestamps <= val_time
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    train_nodes = set(train_data.sources) | set(train_data.destinations)

    # New-node validation: endpoint not seen in train
    val_sources = sources[val_mask]
    val_destinations = destinations[val_mask]
    val_edge_contains_new = np.array(
      [(a not in train_nodes) or (b not in train_nodes) for a, b in zip(val_sources, val_destinations)]
    )
    new_node_val_mask = val_mask.copy()
    new_node_val_mask[val_mask] = val_edge_contains_new

    # Nodes seen up to the end of validation (train + all val edges)
    val_nodes = set(val_sources) | set(val_destinations)
    seen_before_test = train_nodes | val_nodes

    # New-node test: endpoint not seen in train nor in val
    test_sources = sources[test_mask]
    test_destinations = destinations[test_mask]
    test_edge_contains_new = np.array(
      [(a not in seen_before_test) or (b not in seen_before_test)
       for a, b in zip(test_sources, test_destinations)]
    )
    new_node_test_mask = test_mask.copy()
    new_node_test_mask[test_mask] = test_edge_contains_new

  else:
    raise ValueError(f'Unknown new_nodes_mode="{new_nodes_mode}". Use "original implementation" or "real".')

  # Full val/test sets (all edges)
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])
  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # New-node subsets
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
                           labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(
    full_data.n_interactions, full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))

  if new_nodes_mode == "original implementation":
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
      len(new_test_node_set)))

  return edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data
