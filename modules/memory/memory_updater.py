from torch import nn
import torch

class MemoryUpdater(nn.Module):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super().__init__()
    self.memory = memory
    self.layer_norm = nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
      "Trying to update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)
    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.detach().clone(), self.memory.last_update.detach().clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
      "Trying to update memory to time in the past"

    updated_memory = self.memory.memory.detach().clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.detach().clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super().__init__(memory, message_dimension, memory_dimension, device)
    self.memory_updater = nn.GRUCell(input_size=message_dimension, hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  else:
    raise ValueError(f"Memory updater {module_type} not implemented")
