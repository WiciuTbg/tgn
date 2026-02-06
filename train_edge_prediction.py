import math
import time
import numpy as np
import torch
from eval_edge_prediction import eval


def train(
    model,
    pre_test_data,
    train_data,
    val_data,
    new_node_val_data,
    batch_size: int,
    n_epoch: int,
    train_rand_sampler,
    val_rand_sampler,
    optimizer,
    criterion):

    device = model.device

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    for epoch in range(n_epoch):
        start_epoch = time.time()

        # Reinitialize memory of the model at the start of each epoch.
        model.memory.__init_memory__()

        # Train using only the training graph.
        model.set_neighbor_finder(train_data)
        model.train()

        m_loss = []

        for batch_idx in range(num_batch):
            optimizer.zero_grad()

            start_idx = batch_idx * batch_size
            end_idx = min(num_instance, start_idx + batch_size)

            sources_batch = train_data.sources[start_idx:end_idx]
            destinations_batch = train_data.destinations[start_idx:end_idx]
            edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
            timestamps_batch = train_data.timestamps[start_idx:end_idx]

            size = len(sources_batch)
            _, negatives_batch = train_rand_sampler.sample(size)

            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negatives_batch,
                timestamps_batch,
                edge_idxs_batch)

            loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # Detach memory after each batch to avoid backprop through the entire timeline.
            model.memory.detach_memory()

        epoch_times.append(time.time() - start_epoch)

        # Validation uses the full graph.
        model.set_neighbor_finder(pre_test_data)
        model.eval()

        # Backup memory at end of training epoch for unseen-node validation.
        train_memory_backup = model.memory.backup_memory()

        val_ap, val_auc = eval(
            model=model,
            negative_edge_sampler=val_rand_sampler,
            data=val_data)

        val_memory_backup = model.memory.backup_memory()

        # Restore training memory to evaluate on unseen nodes.
        model.memory.restore_memory(train_memory_backup)

        nn_val_ap, nn_val_auc = eval(
            model=model,
            negative_edge_sampler=val_rand_sampler,
            data=new_node_val_data)

        # Restore memory after validation for potential testing later.
        model.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        train_losses.append(float(np.mean(m_loss)))

        total_epoch_times.append(time.time() - start_epoch)
        
        print(f"epoch: {epoch} took {total_epoch_times[-1]:.2f}s", flush=True)
        print(f"Epoch mean loss: {train_losses[-1]:.6f}", flush=True)
        print(f"val auc: {val_auc:.6f}, new node val auc: {nn_val_auc:.6f}", flush=True)
        print(f"val ap: {val_ap:.6f}, new node val ap: {nn_val_ap:.6f}", flush=True)

    return model
