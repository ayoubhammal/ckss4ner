# implementation of https://aclanthology.org/2020.emnlp-main.516.pdf
# this script contains the 2 algorims described in the paper: NNShot and StructShot
# the algorithms use a bert model pretrained on ontonotes5 NER task
# then, samples a support set from the dev set of the evaluation dataset to simulate a few-shot scenario
# and evaluates on the whole test set

# evaluation tasks:
#   tag set extension - specific to ontonotes5, will be left out for now
#   domain transfer

import argparse
import sys
import io
import time
import random
import itertools
import json
import logging
import os
import pathlib

import torch
import numpy as np
import transformers
from ratio_constraints.optim import mAdamW

from ratio_constraints.network import NetworkForTokenClassification
from ratio_constraints.utils import IOTagDict, batchify, TagDict, get_abstract_transitions, greedy_support_set_sampling, project_target_transitions
import ratio_constraints.repr
import ratio_constraints.io
from ratio_constraints.loss_builders import tagging_loss_builder_factory
from ratio_constraints.decoders import TaggingDecoder
from ratio_constraints.eval import NEREvaluator


cmd = argparse.ArgumentParser()

cmd.add_argument("--model", type=str, default=None, required=False)
cmd.add_argument("--pretrain-train", type=str, required=True)
cmd.add_argument("--supports", type=str, required=True)
cmd.add_argument("--test", type=str, required=True)
cmd.add_argument("--target-set-tags-file", type=str, required=True)
cmd.add_argument("--tau", type=float, required=True)

cmd.add_argument("--eval-minibatch-size", type=int, default=300)

cmd.add_argument("--device", type=str, default="cpu")
cmd.add_argument("--storage-device", type=str, required=False, default="cpu")

cmd.add_argument('--bert-tokenizer', type=str)
cmd.add_argument('--bert-model', type=str)
cmd.add_argument('--bert-do-lower-case', action="store_true")
cmd.add_argument('--bert-start-features', action="store_true")
cmd.add_argument('--bert-end-features', action="store_true")
cmd.add_argument('--bert-proxy', type=str, default="")
cmd.add_argument('--bert-cache', type=str, default="")

cmd.add_argument("--dropout", type=float, default=0.5)

cmd.add_argument("--seed", type=int, default=0)

args = cmd.parse_args()

if args.seed != 0:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# train data is used to calculate each transition frequency
train_data = ratio_constraints.io.read(args.pretrain_train)
train_tag_dict = TagDict(
    set(
        itertools.chain(*[sentence["tags"] for sentence in train_data])
    ).union({"O"})  # O may not exist in partially labeled training datasets
)
# dev data will be sampled into different support sets
# each support set will be translated into IO format
test_data = ratio_constraints.io.read(args.test)

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
# the tag dict in IO format used for:
#   - support data processing (useless, we only need the string tags)
#   - build the transitions (usefull, we need to know the number of IO tags)
#   - build the transition and begining masks (usefull, nedded for the decoding)
target_io_tag_dict = target_tag_dict.to_io_tag_dict()

bert_input_builder = ratio_constraints.repr.BertInputBuilder(
    tokenizer_name=args.bert_tokenizer,
    proxy=args.bert_proxy,
    cache=args.bert_cache,
    start_features=args.bert_start_features,
    end_features=args.bert_end_features,
    do_lower_case=args.bert_do_lower_case
)

# construct entity type to sentence index on the dev set
# this index takes into account entity types appearing multiple times in one sentence
# in this case this sentence will be repeated as many times in the index
supports_data = []
for support_set_path in pathlib.Path(args.supports).iterdir():
    support_data = ratio_constraints.io.read(support_set_path)
    # transform the tags to IO format
    support_data = [
        {
            "tokens": datum["tokens"],
            "tags": datum["tags"],
            "io_tags": IOTagDict.bio_to_io(datum["tags"])
        } for datum in support_data
    ]
    ratio_constraints.repr.build_torch_tensors(
        support_data,
        bert_input_builder,
        target_tag_dict,
        device=args.storage_device
    )
    support_data = batchify(
        support_data,
        args.eval_minibatch_size,
        lambda sentence: 1
    )
    supports_data.append(support_data)

ratio_constraints.repr.build_torch_tensors(
    test_data,
    bert_input_builder,
    target_tag_dict,
    device=args.storage_device
)
test_data = batchify(
    test_data,
    args.eval_minibatch_size,
    lambda sentence: 1
)

# their transiition weights contain the transition start -> tag and tag -> start
# which we do not need
transition_weights, beginning_mask = project_target_transitions(target_io_tag_dict, get_abstract_transitions(train_data), args.tau)
transition_weights = transition_weights.to(args.device)
beginning_mask = beginning_mask.to(args.device)

transition_mask = target_io_tag_dict.build_transition_mask(dtype=torch.float, device=args.device)
# beginning_mask = target_io_tag_dict.build_beginning_mask(dtype=torch.float, device=args.device)
decoder = TaggingDecoder(transition_mask=transition_mask, beginning_mask=beginning_mask)

nnshot_evaluator = NEREvaluator()
structshot_evaluator = NEREvaluator()

network = NetworkForTokenClassification(
    args.bert_model,
    len(train_tag_dict),
    dropout=args.dropout,
    bert_cache=args.bert_cache,
    bert_proxy=args.bert_proxy
)
network.to(args.device)

if args.model is not None:
    checkpoint = torch.load(args.model)
    network.bert_load_state_dict(checkpoint["model_state_dict"])

torch.set_grad_enabled(False)
network.eval()

all_nnshot_results = {
    "precision": [],
    "recall": [],
    "f1": [],
}
all_structshot_results = {
    "precision": [],
    "recall": [],
    "f1": [],
}
for i_support, support_data in enumerate(supports_data):
    print("support n {}\n".format(i_support), file=sys.stderr, flush=True)
    print("support n {}\n".format(i_support), flush=True)
    nnshot_evaluator.reset()
    structshot_evaluator.reset()

    # extract embeddings of the support set
    features_per_tag = {tag: [] for tag in target_io_tag_dict.id_to_tag}
    for batch_data in support_data:
        bert_repr = [
            {
                k:
                    (v.to(args.device) if k != "n_tokens" else v)
                for k, v in t["bert_repr"].items()
            }
            for t in batch_data
        ]
        batch_features = network.get_features(bert_repr) # (n * l_i, h)

        i = 0
        for t in batch_data:
            for tag in t["io_tags"]:
                features_per_tag[tag].append(batch_features[i])
                i += 1

    # for each tag, we have a set of token embeddings that belong to that tag
    features_per_tag = {tag: torch.vstack(features) for tag, features in features_per_tag.items()}


    # extract embeddings of test_set
    for batch_data in test_data:
        bert_repr = [
            {
                k:
                    (v.to(args.device) if k != "n_tokens" else v)
                for k, v in t["bert_repr"].items()
            }
            for t in batch_data
        ]
        batch_features = network.get_features(bert_repr) # (n * l_i, h)

        # collect the token embeddings of the test batch
        test_set_features = [] # (n, l_i, h)
        start = 0
        for b, size in enumerate(sentence["n_tokens"] for sentence in bert_repr):
            test_set_features.append(batch_features[start: start + size, :])
            start += size


        batch_emissions = []
        for f, tags in zip(test_set_features, (sentence["tags"] for sentence in batch_data)):
            # nnshot
            # the emission for a token and a cluster (tag) is the distance between that token and the closest point in that cluster
            distances = torch.vstack(
                # (l_i, c, d) -> (l_i, c)
                [((f.unsqueeze(1) - features_per_tag[tag].unsqueeze(0))**2).sum(axis=2).min(dim=1).values for tag in target_io_tag_dict.tag_to_id],
            ).T
            nnshot_emissions = torch.log_softmax(-distances, dim=1)
            # nnshot_emissions = -distances
            nnshot_prediction = nnshot_emissions.argmax(dim=1).to("cpu") # (l, )

            nnshot_evaluator(tags, IOTagDict.io_to_bio(target_io_tag_dict.idx_to_tags(nnshot_prediction)))

            # save the emissions for structshot prediction
            batch_emissions.append(nnshot_emissions)

        batch_emissions = torch.nn.utils.rnn.pad_sequence(
            batch_emissions,
            batch_first=True,
            padding_value=0 # the padding value should not matter since we have the lengths passed to the decoder
        )

        # structshot
        torch.set_grad_enabled(True)
        structshot_predictions = decoder(batch_emissions, transition_weights, [len(sentence["tokens"]) for sentence in batch_data])
        torch.set_grad_enabled(False)
        for sentence, pred in zip(batch_data, structshot_predictions):
            structshot_evaluator(sentence["tags"], IOTagDict.io_to_bio(target_io_tag_dict.idx_to_tags(pred)))


    nnshot_results = {
        "precision": nnshot_evaluator.precision() * 100,
        "recall": nnshot_evaluator.recall() * 100,
        "f1": nnshot_evaluator.f1() * 100,
    }
    structshot_results = {
        "precision": structshot_evaluator.precision() * 100,
        "recall": structshot_evaluator.recall() * 100,
        "f1": structshot_evaluator.f1() * 100,
    }

    print("nnshot results: {}".format(nnshot_results))
    print("structshot results: {}\n".format(structshot_results))

    for k in all_nnshot_results.keys():
        all_nnshot_results[k].append(nnshot_results[k])
        all_structshot_results[k].append(structshot_results[k])

print("final nnshot results over {} support sets:".format(len(supports_data)))
print("    precision: {} +- {}".format(np.mean(all_nnshot_results["precision"]), np.std(all_nnshot_results["precision"])))
print("    recall   : {} +- {}".format(np.mean(all_nnshot_results["recall"]), np.std(all_nnshot_results["recall"])))
print("    f1       : {} +- {}".format(np.mean(all_nnshot_results["f1"]), np.std(all_nnshot_results["f1"])))

print("final structshot results over {} support sets:".format(len(supports_data)))
print("    precision: {} +- {}".format(np.mean(all_structshot_results["precision"]), np.std(all_structshot_results["precision"])))
print("    recall   : {} +- {}".format(np.mean(all_structshot_results["recall"]), np.std(all_structshot_results["recall"])))
print("    f1       : {} +- {}".format(np.mean(all_structshot_results["f1"]), np.std(all_structshot_results["f1"])))

print("{} +- {} | {} +- {}".format(np.mean(all_nnshot_results["f1"]), np.std(all_nnshot_results["f1"]), np.mean(all_structshot_results["f1"]), np.std(all_structshot_results["f1"])), flush=True)
