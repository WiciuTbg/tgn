import math
import logging
import time
import pickle
from pathlib import Path
import numpy as np
import torch

from evaluation.evaluation import eval_edge_prediction
from utils.others import EarlyStopMonitor


def train(
    model,
    train_data,
    val_data,
    new_node_val_data,
    train_ngh_finder,
    full_ngh_finder,
    batch_size: int,
    n_epoch: int,
    train_rand_sampler,
    val_rand_sampler,
    optimizer,
    criterion,
    device,
    num_neighbors: int,
    results_path: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_prefix: str | None = None,
    logger: logging.Logger | None = None,
    patience: int = 5,
):
    # Use a provided logger, otherwise fall back to a module logger.
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create output directories only if saving is enabled.
    if results_path is not None:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(epoch: int) -> str:
        # If checkpoint saving is disabled, this should not be called.
        assert checkpoint_dir is not None and checkpoint_prefix is not None
        return str(Path(checkpoint_dir) / f"{checkpoint_prefix}-{epoch}.pth")

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    logger.info("num of training instances: %d", num_instance)
    logger.info("num of batches per epoch: %d", num_batch)

    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=patience)

    for epoch in range(n_epoch):
        start_epoch = time.time()

        # Reinitialize memory of the model at the start of each epoch.
        model.memory.__init_memory__()

        # Train using only the training graph.
        model.set_neighbor_finder(train_ngh_finder)
        model.train()

        m_loss = []
        logger.info("start epoch %d", epoch)

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
                edge_idxs_batch,
                num_neighbors,
            )

            loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # Detach memory after each batch to avoid backprop through the entire timeline.
            model.memory.detach_memory()

        epoch_times.append(time.time() - start_epoch)

        # Validation uses the full graph.
        model.set_neighbor_finder(full_ngh_finder)
        model.eval()

        # Backup memory at end of training epoch for unseen-node validation.
        train_memory_backup = model.memory.backup_memory()

        val_ap, val_auc = eval_edge_prediction(
            model=model,
            negative_edge_sampler=val_rand_sampler,
            data=val_data,
            n_neighbors=num_neighbors,
        )

        val_memory_backup = model.memory.backup_memory()

        # Restore training memory to evaluate on unseen nodes.
        model.memory.restore_memory(train_memory_backup)

        nn_val_ap, nn_val_auc = eval_edge_prediction(
            model=model,
            negative_edge_sampler=val_rand_sampler,
            data=new_node_val_data,
            n_neighbors=num_neighbors,
        )

        # Restore memory after validation for potential testing later.
        model.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        train_losses.append(float(np.mean(m_loss)))

        total_epoch_times.append(time.time() - start_epoch)

        logger.info("epoch: %d took %.2fs", epoch, total_epoch_times[-1])
        logger.info("Epoch mean loss: %.6f", train_losses[-1])
        logger.info("val auc: %.6f, new node val auc: %.6f", val_auc, nn_val_auc)
        logger.info("val ap: %.6f, new node val ap: %.6f", val_ap, nn_val_ap)

        # Save temporary results.
        if results_path is not None:
            with open(results_path, "wb") as f:
                pickle.dump(
                    {
                        "val_aps": val_aps,
                        "new_nodes_val_aps": new_nodes_val_aps,
                        "train_losses": train_losses,
                        "epoch_times": epoch_times,
                        "total_epoch_times": total_epoch_times,
                    },
                    f,
                )

        # Early stopping + checkpointing.
        if early_stopper.early_stop_check(val_ap):
            logger.info("No improvement over %d epochs, stop training", early_stopper.max_round)

            if checkpoint_dir is not None and checkpoint_prefix is not None:
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                logger.info("Loading best model from epoch %d: %s", early_stopper.best_epoch, best_model_path)
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                model.eval()
            break
        else:
            if checkpoint_dir is not None and checkpoint_prefix is not None:
                torch.save(model.state_dict(), get_checkpoint_path(epoch))

    return model
