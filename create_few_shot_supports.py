# implementation of https://aclanthology.org/2020.emnlp-main.516.pdf

import argparse
import random
import pathlib
import os

import torch
import numpy as np

from ratio_constraints.utils import TagDict, greedy_support_set_sampling
import ratio_constraints.io


cmd = argparse.ArgumentParser()

cmd.add_argument("--data", type=str, required=True)
cmd.add_argument("--output", type=str, required=True)
cmd.add_argument("--target-set-tags-file", type=str, required=True)

cmd.add_argument("--k-shots", type=int, required=True)
cmd.add_argument("--n-support-sets", type=int, required=True)
cmd.add_argument("--use-bio-tags", default=False, action="store_true")

cmd.add_argument("--seed", type=int, default=0)

args = cmd.parse_args()


# create the output path if does not exist
pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)


if args.seed != 0:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

data = ratio_constraints.io.read(args.data)

with open(args.target_set_tags_file, "r") as target_set_tag_file:
    target_tag_set = set(
        filter(
            lambda tag: tag != "",
            map(
                lambda tag: tag.strip(),
                target_set_tag_file.readlines()
            )
        )
    )

# the tag dict in BIO format used for:
#   - support set sampling (usefull, to get the list of tags)
#   - test data processing (useless, we need only the string tags)
#   - model initialization (useless, we need only the bert part)
target_tag_dict = TagDict(target_tag_set)

# construct entity type to sentence index on the dev set
# this index takes into account entity types appearing multiple times in one sentence
# in this case this sentence will be repeated as many times in the index
entity_type_index = {
    entity_type: set()
    for entity_type in {tag if args.use_bio_tags else tag[2:] for tag in target_tag_dict.id_to_tag if tag != "O"}
}
for i_datum, datum in enumerate(data):
    for tag in datum["tags"]:
        if args.use_bio_tags:
            if tag != "O":
                entity_type_index[tag].add(i_datum)
        else:
            if tag.startswith("B-"):
                entity_type_index[tag.removeprefix("B-")].add(i_datum)

support_sets = greedy_support_set_sampling(entity_type_index, data, args.k_shots, args.n_support_sets, args.use_bio_tags)

for i, support_set in enumerate(support_sets):
    support_set = [
        {
            "tokens": datum["tokens"],
            "tags": datum["tags"],
        } for datum in support_set
    ]
    
    output_path = os.path.join(args.output, "entity.support_k={}_{}.txt".format(args.k_shots, i))
    with open(output_path, "w") as file:
        for datum in support_set:
            file.write(" ".join(datum["tokens"]) + "\n")
            file.write(" ".join(datum["tags"]) + "\n\n")

