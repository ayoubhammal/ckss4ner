# This script compares our implementation with the results of torch_struct,
# so we are sure we didn't make any mistake...
# If no error is raised when running this script,
# this means that our implementation computes the same values as torch_struct
#
# Note: torch_struct does not support sentences of length 1

import torch
from torch_struct import LinearChainCRF
import ratio_constraints.tagging_algorithms


def convert_to_torch_struct_potentials(tag_weights, transition_weights):
    num_words = tag_weights.shape[1]
    num_tags = tag_weights.shape[2]

    potentials = (
            tag_weights[:, 1:num_words].unsqueeze(3)
            +
            # transposed compared to what I used in my code above
            transition_weights.t().reshape(1, 1, num_tags, num_tags)
    )
    potentials[:, 0] += tag_weights[:, 0].unsqueeze(1)

    return potentials


num_tags = 3
lengths = [5, 4, 3]
max_length = max(lengths)
batch_size = len(lengths)

tag_weights = (torch.rand(batch_size, max_length, num_tags) - 0.5) * 5
transition_weights = (torch.rand(num_tags, num_tags) - 0.5) * 5

ts_tag_weights = tag_weights.clone().detach().requires_grad_(True)
torch_struct_potentials = convert_to_torch_struct_potentials(ts_tag_weights, transition_weights)
model = LinearChainCRF(torch_struct_potentials, torch.LongTensor(lengths))
ts_log_partitions = model.partition.clone().detach()
model.partition.backward(torch.ones_like(ts_log_partitions))
ts_marginals = ts_tag_weights.grad.clone().detach()
ts_max = model.max

custom_log_partitions = ratio_constraints.tagging_algorithms.tagging_log_partition(tag_weights, transition_weights, lengths)
custom_marginals, _ = ratio_constraints.tagging_algorithms.tagging_marginals(tag_weights, transition_weights, lengths)
# we don't test argmax... let's just hope it works :)
custom_max, _ = ratio_constraints.tagging_algorithms.tagging_argmax(tag_weights, transition_weights, lengths)

assert torch.allclose(ts_log_partitions, custom_log_partitions)
assert torch.allclose(ts_marginals, custom_marginals)
assert torch.allclose(custom_max, ts_max)
