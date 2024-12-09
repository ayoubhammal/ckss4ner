from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Set

import numpy as np
import torch


class TagDict:
    def __init__(self, tags, unk="X"):
        self.unk = unk

        self.tag_to_id = dict()
        self.id_to_tag = list()

        for t in set(tags):
            if t != unk:
                self.tag_to_id[t] = len(self.id_to_tag)
                self.id_to_tag.append(t)

        assert all((t[:2] != "I-" or ("B-" + t[2:]) in self.tag_to_id) for t in self.tag_to_id.keys())

    def __len__(self):
        return len(self.id_to_tag)

    def build_transition_mask(self, dtype=torch.float, device="cpu"):
        mask = torch.zeros(len(self), len(self), dtype=dtype, device="cpu")  # moved to correct device at the end

        prevs = defaultdict(lambda: list())
        for idx, tag in enumerate(self.id_to_tag):
            if tag.startswith("I-"):
                prevs[idx].append(idx)  # transition I -> I
                prevs[idx].append(self.tag_to_id["B" + tag[1:]])  # transition B -> I

        for idx, allowed_prev_idx in prevs.items():
            mask[:, idx] = float("-inf")
            for prev_idx in allowed_prev_idx:
                mask[prev_idx, idx] = 0

        return mask.to(device)

    def build_beginning_mask(self, dtype=torch.float, device="cpu"):
        mask = torch.zeros(len(self), dtype=dtype, device="cpu")  # moved to correct device at the end

        for idx, tag in enumerate(self.id_to_tag):
            if tag.startswith("I-"):
                mask[idx] = float("-inf")

        return mask.to(device)

    def tags_to_idx(self, tags):
        return [(self.tag_to_id[t] if t != self.unk else -1) for t in tags]

    def idx_to_tags(self, tags):
        assert all(t >= 0 for t in tags)
        return [self.id_to_tag[t] for t in tags]

    def to_io_tag_dict(self):
        return IOTagDict(
            set(tag for tag in self.id_to_tag if not tag.startswith("B-")),
            self.unk
        )

class IOTagDict:
    def __init__(self, tags, unk="X"):
        self.unk = unk

        self.tag_to_id = dict()
        self.id_to_tag = list()

        for t in set(tags):
            if t != unk:
                self.tag_to_id[t] = len(self.id_to_tag)
                self.id_to_tag.append(t)

    def __len__(self):
        return len(self.id_to_tag)

    def build_transition_mask(self, dtype=torch.float, device="cpu"):
        return torch.zeros(len(self), len(self), dtype=dtype, device=device)

    def build_beginning_mask(self, dtype=torch.float, device="cpu"):
        return torch.zeros(len(self), dtype=dtype, device=device)

    def tags_to_idx(self, tags):
        return [(self.tag_to_id[t] if t != self.unk else -1) for t in tags]

    def idx_to_tags(self, tags):
        assert all(t >= 0 for t in tags)
        return [self.id_to_tag[t] for t in tags]

    @staticmethod
    def bio_to_io(tags):
        return ["I-" + tag[2:] if tag.startswith("B-") else tag for tag in tags]

    @staticmethod
    def io_to_bio(tags):
        bio_tags = []

        i = 0
        while i < len(tags):
            if tags[i] != "O":
                entity_type = tags[i].removeprefix("I-")
                bio_tags.append("B-" + entity_type)
                j = i + 1
                while j < len(tags) and tags[j] == "I-" + entity_type:
                    bio_tags.append(tags[j])
                    j += 1
                i = j
            else:
                bio_tags.append(tags[i])
                i += 1
        return bio_tags

def batchify(data, max_size, key=lambda x: len(x)):
    data = sorted(data, key=key, reverse=True)

    batchified = list()
    current_size = 0
    for sentence in data:
        if current_size == 0 or current_size + key(sentence) > max_size:
            batchified.append(list())
            current_size = 0
        batchified[-1].append(sentence)
        current_size += key(sentence)

    return batchified


def bio_to_mentions(tags):
    mentions = list()

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            mentions.append([tag[2:], i, i])
        elif tag.startswith("I-"):
            assert i > 0
            assert mentions[-1][0] == tag[2:]
            mentions[-1][2] = i
        else:
            assert tag == "O"

    return set(tuple(m) for m in mentions)

def greedy_support_set_sampling(entity_type_index: Dict[str, Set[int]], dataset: List[Dict[str, List[str]]], k_shots: int, n_support_sets: int = 1, use_bio_tags: bool = False) -> List[List[Dict[str, List[str]]]]:
    support_sets = []

    sorted_entity_types = sorted(entity_type_index.keys(), key=lambda entity_type: len(entity_type_index[entity_type]))

    rng = np.random.default_rng()

    for s in range(n_support_sets):
        entity_type_index_ = deepcopy(entity_type_index)
        support_set = []

        counts = {entity_type: 0 for entity_type in sorted_entity_types}

        for entity_type in sorted_entity_types:
            while counts[entity_type] < k_shots and len(entity_type_index_[entity_type]) > 0:
                # sample a santence with this entity type
                sample_sentence_id: int = rng.choice(list(entity_type_index_[entity_type]), size=1, replace=False, shuffle=False).item()
                # remove this sentences to not be sampled again
                entity_type_index_ = {et: [index for index in indices if index != sample_sentence_id] for et, indices in entity_type_index_.items()}

                sample_sentence = dataset[sample_sentence_id]
                support_set.append(sample_sentence)

                # update the counts
                for tag in sample_sentence["tags"]:
                    if use_bio_tags:
                        if tag != "O":
                            counts[tag] += 1
                    else:
                        if tag.startswith("B-"):
                            counts[tag.removeprefix("B-")] += 1

        support_sets.append(support_set)

        print("{} counts:\n{}".format(s, counts))

    assert len(support_sets) == n_support_sets

    return support_sets

# https://github.com/asappresearch/structshot/blob/main/structshot/run_pl_pred.py#L224
def get_abstract_transitions(data):
    """
    Compute abstract transitions on the training dataset for StructShot
    """
    tag_lists = [example["tags"] for example in data]

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans

# https://github.com/asappresearch/structshot/blob/main/structshot/viterbi.py#L22
def project_target_transitions(io_tag_dict, abstract_transitions, tau):

    O_ID = io_tag_dict.tag_to_id["O"]
    n_tag = len(io_tag_dict) + 1
    START_ID = n_tag - 1

    i_mask = ~np.isin(np.arange(n_tag), [O_ID, START_ID])

    s_o, s_i, o_o, o_i, i_o, i_i, x_y = abstract_transitions

    # self transitions for I-X tags
    a = torch.eye(n_tag) * i_i
    # transitions from I-X to I-Y
    b = torch.ones(n_tag, n_tag) * x_y / (n_tag - 3)
    c = torch.eye(n_tag) * x_y / (n_tag - 3)

    transitions = a + b - c
    # transition from START to O
    transitions[START_ID, O_ID] = s_o
    # transitions from START to I-X
    transitions[START_ID, i_mask] = s_i / (n_tag - 2)
    # transition from O to O
    transitions[O_ID, O_ID] = o_o
    # transitions from O to I-X
    transitions[O_ID, i_mask] = o_i / (n_tag - 2)
    # transitions from I-X to O
    transitions[i_mask, O_ID] = i_o
    # no transitions to START
    transitions[:, START_ID] = 0.

    powered = torch.pow(transitions, tau)
    summed = powered.sum(dim=1)

    transitions = powered / summed.view(n_tag, 1)

    transitions = torch.log(torch.where(transitions > 0, transitions, torch.tensor(.000001)))[:, :-1]

    begin_transitions = transitions[-1, :]
    transitions = transitions[:-1, :]
    return transitions, begin_transitions
