import argparse
import sys
import time
import random
from copy import deepcopy
import pathlib

import torch
import numpy as np

from ratio_constraints.network import NetworkForTokenClassification
from ratio_constraints.utils import batchify, TagDict
import ratio_constraints.repr
import ratio_constraints.io
from ratio_constraints.decoders import TaggingDecoder
from ratio_constraints.eval import NEREvaluator
from ratio_constraints.clustering import LDAKMeans


cmd = argparse.ArgumentParser()

cmd.add_argument("--model", type=str, required=False)

cmd.add_argument("--train", type=str, required=True)
cmd.add_argument("--supports", type=str, required=True)
cmd.add_argument("--dev", type=str, required=True)
cmd.add_argument("--test", type=str, required=True)
cmd.add_argument("--target-set-tags-file", type=str, required=True)
cmd.add_argument("--tau", type=float, required=True)

cmd.add_argument("--n-o-clusters", type=int, required=True)
cmd.add_argument("--n-i-clusters", type=int, required=True)
cmd.add_argument("--feature-weights", type=str, default=None)
cmd.add_argument("--s", type=float, default=None)
cmd.add_argument("--temp", type=float, default=None)
cmd.add_argument("--n-iter", type=int, required=True)
cmd.add_argument("--kmeans-n-iter", type=int, required=True)
cmd.add_argument("--n-iter-bregman", type=int, default=20)
cmd.add_argument("--assignments", choices=["soft", "hard"], required=True)
cmd.add_argument("--centroids-init", choices=["random", "k-means++", "means", "hierarchical-ward", "hierarchical-complete", "hierarchical-average", "hierarchical-single"], required=True)

cmd.add_argument("--dont-use-train", default=False, action="store_true")
cmd.add_argument("--dont-use-dev", default=False, action="store_true")

cmd.add_argument("--eval-minibatch-size", type=int, default=300)
cmd.add_argument("--o-tag-ratio", type=float, default=None)

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


def get_features(batch_data, network, device, tag_dict=None):
    bert_repr = [
        {
            k:
                (v.to(device) if k != "n_tokens" else v)
            for k, v in t["bert_repr"].items()
        }
        for t in batch_data
    ]
    batch_features = network.get_features(bert_repr).detach().cpu().numpy()

    output_features = []
    output_tags = []
    output_gold_tags = []
    output_tokens = []
    start = 0
    for bd in batch_data:
        output_features.append(batch_features[start: start + bd["bert_repr"]["n_tokens"]])
        start += bd["bert_repr"]["n_tokens"]

        output_tags.append(bd["tags_tensor"].detach().cpu().numpy())

        if tag_dict is not None:
            output_gold_tags.append(np.asarray(tag_dict.tags_to_idx(bd["gold_tags"])))

        output_tokens.append(bd["tokens"])

    if tag_dict is not None:
        return output_features, output_tags, output_tokens, output_gold_tags
    return output_features, output_tags, output_tokens


def make_evaluate(evaluator, decoder, transition_weights, tag_dict, cluster2tag):
    def evaluate(kmeans, features, full_tags):
        evaluator.reset()

        accuracy = 0
        cpt = 0

        cpt_good_i = 0
        cpt_i = 0
        for i_batch in range(0, len(features), args.eval_minibatch_size):
            batch_features = features[i_batch: i_batch + args.eval_minibatch_size]
            batch_tags = full_tags[i_batch: i_batch + args.eval_minibatch_size]
            batch_n_tokens = list(map(len, batch_features))


            # calculate the emissions per cluster
            batch_emissions_per_cluster = [torch.from_numpy(f).to(args.device) for f in kmeans.predict_emissions(batch_features)]


            # aggregate emissions per cluster to emissions per tag using the maximum
            batch_emissions = []
            for e in batch_emissions_per_cluster:
                emission = torch.full((e.shape[0], len(tag_dict)), -np.inf, device=args.device)
                
                for cluster, tag in cluster2tag.items():
                    emission[:, tag] = torch.maximum(emission[:, tag], e[:, cluster])

                batch_emissions.append(emission)

            # calculate plain accuracy
            for e, gold_tags in zip(batch_emissions, batch_tags):
                predictions = e.argmax(axis=-1).detach().cpu().numpy()

                predictions_tags = tag_dict.idx_to_tags(predictions)

                for i, pred_tag in enumerate(predictions_tags):
                    if pred_tag.startswith("I-"):
                        cpt_i += 1
                        if i > 0 and predictions_tags[i - 1] in ["B-" + pred_tag.removeprefix("I-"), pred_tag]:
                            cpt_good_i += 1

                cpt += len(gold_tags)
                accuracy += (predictions == gold_tags).sum()
            
            batch_emissions = torch.nn.utils.rnn.pad_sequence(
                batch_emissions,
                batch_first=True,
                padding_value=0 # the padding value should not matter since we have the lengths passed to the decoder
            )

            # viterbi decoding
            torch.set_grad_enabled(True)
            _predictions = decoder(batch_emissions, transition_weights, batch_n_tokens)
            torch.set_grad_enabled(False)
            
            for gold_tags, pred in zip(batch_tags, _predictions):
                evaluator(tag_dict.idx_to_tags(gold_tags), tag_dict.idx_to_tags(pred))

        accuracy = accuracy / cpt

        return evaluator.f1(), {"accuracy": accuracy, "good I ratio": cpt_good_i / cpt_i}
    return evaluate


# setting random seed
if args.seed != 0:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


# reading target tag list for evaluation
with open(args.target_set_tags_file, "r") as target_set_tag_file:
    tag_set = set(
        filter(
            lambda tag: tag != "",
            map(
                lambda tag: tag.strip(),
                target_set_tag_file.readlines()
            )
        )
    )
tag_dict = TagDict(
    tag_set.union({"O"})  # O may not exist in partially labeled training datasets
)


# reading the datasets
train_data = ratio_constraints.io.read(args.train)
dev_data = ratio_constraints.io.read(args.dev)
test_data = ratio_constraints.io.read(args.test)

supports_data = []
for support_set_path in pathlib.Path(args.supports).iterdir():
    supports_data.append(ratio_constraints.io.read(support_set_path))


train_data = [
    {
        "tokens": d["tokens"],
        "tags": ["X"] * len(d["tags"]),
        "gold_tags": d["tags"]
    }
    for d in train_data
]
dev_data = [
    {
        "tokens": d["tokens"],
        "tags": ["X"] * len(d["tags"]),
        "gold_tags": d["tags"]
    }
    for d in dev_data
]

# writing labels from the supports data
train_supports_data = []
for support_data in supports_data:
    if not args.dont_use_dev:
        data = deepcopy(dev_data)
        if not args.dont_use_train:
            data += deepcopy(train_data)
        
        for d in support_data:
            found = False
            for du in data:
                if len(d["tokens"]) == len(du["tokens"]) and \
                    all(t == tu for t, tu in zip(d["tokens"], du["tokens"])):
                    # assert all(t == tu for t, tu in zip(d["tags"], du["gold_tags"])), "huh, same tokens different tags?, tokens: {}, tags from dataset: {}, tags from support: {}".format(du["tokens"], du["gold_tags"], d["tags"])
                    
                    found = True
                    du["tags"] = d["tags"]
            assert found, "could not find this support sentence in train or dev: {}".format(d)
    else:
        data = deepcopy(support_data)
        for d in data:
            d["gold_tags"] = d["tags"]
            d["tags"] = d["tags"]

    train_supports_data.append(data)

bert_input_builder = ratio_constraints.repr.BertInputBuilder(
    tokenizer_name=args.bert_tokenizer,
    proxy=args.bert_proxy,
    cache=args.bert_cache,
    start_features=args.bert_start_features,
    end_features=args.bert_end_features,
    do_lower_case=args.bert_do_lower_case
)

for data in train_supports_data:
    ratio_constraints.repr.build_torch_tensors(
        data,
        bert_input_builder,
        tag_dict,
        device=args.storage_device,
        build_tags_tensor=True
    )
ratio_constraints.repr.build_torch_tensors(
    test_data,
    bert_input_builder,
    tag_dict,
    device=args.storage_device,
    build_tags_tensor=True
)

train_supports_data = [
    batchify(
        data,
        args.eval_minibatch_size,
        lambda sentence: 1
    )
    for data in train_supports_data
]
test_data = batchify(
    test_data,
    args.eval_minibatch_size,
    lambda sentence: 1
)


# loading the model and its checkpoint
network = NetworkForTokenClassification(
    args.bert_model,
    len(tag_dict),
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

# evaluators
evaluator = NEREvaluator()

# transition matrix and mask
transition_weights = torch.zeros((len(tag_dict), len(tag_dict)), device=args.device, requires_grad=False)

transition_mask = tag_dict.build_transition_mask(dtype=torch.float, device=args.device)
beginning_mask = tag_dict.build_beginning_mask(dtype=torch.float, device=args.device)
decoder = TaggingDecoder(transition_mask=transition_mask, beginning_mask=beginning_mask)


# create cluster 2 tag dictionary
# currently, only the O tag can have multiple clusters
cluster2tag = dict()
i = 0
for tag, tag_id in tag_dict.tag_to_id.items():
    if tag == "O":
        n_clusters = args.n_o_clusters
    else:
        n_clusters = args.n_i_clusters

    for _ in range(n_clusters):
        cluster2tag[i] = tag_id
        i += 1

tag2clusters = {
    i: [cluster for cluster, tag in cluster2tag.items() if tag == i]
    for i in range(len(tag_dict))
}



print("calculating test embeddings... ", end="", file=sys.stderr, flush=True)
forward_start_time = time.time()
test_features, test_tags, test_tokens = list(zip(*[get_features(batch_data, network, args.device) for batch_data in test_data]))

test_features = sum(test_features, [])
test_tags = sum(test_tags, [])
test_tokens = sum(test_tokens, [])

print("{} (s)".format(time.time() - forward_start_time), file=sys.stderr, flush=True)
print("--------------------------------------------------------------------------\n", file=sys.stderr, flush=True)




all_results = {
    "precision": [],
    "recall": [],
    "f1": [],
}

for i_support, (support_data, train_support_data) in enumerate(zip(supports_data, train_supports_data)):
    print("support n {}\n".format(i_support), file=sys.stderr, flush=True)
    print("support n {}\n".format(i_support), flush=True)
    print("--------------------------------------------------------------------------", file=sys.stderr, flush=True)
    print("calculating train embeddings... ", end="", file=sys.stderr, flush=True)
    forward_start_time = time.time()
    train_features, train_tags, train_tokens, train_gold_tags = list(zip(*[get_features(batch_data, network, args.device, tag_dict) for batch_data in train_support_data]))

    train_features = sum(train_features, [])
    train_tags = sum(train_tags, [])
    train_tokens = sum(train_tokens, [])
    train_gold_tags = sum(train_gold_tags, [])

    print("{} (s)\n".format(time.time() - forward_start_time), file=sys.stderr, flush=True)

    print("--------------------------------------------------------------------------", file=sys.stderr, flush=True)
    print("number of o clusters is {} cluster(s)\n".format(args.n_o_clusters), file=sys.stderr, flush=True)
    print("number of i clusters is {} cluster(s)\n".format(args.n_i_clusters), file=sys.stderr, flush=True)
    print("o tag ratio is {}\n".format(args.o_tag_ratio), file=sys.stderr, flush=True)
    print("--------------------------------------------------------------------------\n", file=sys.stderr, flush=True)

    # create the size ratios for each tag
    if args.o_tag_ratio is not None:
        n_total = sum(len(tags) for tags in train_tags)

        cluster_family_sizes = {
            tuple(tag2clusters[tag_dict.tag_to_id["O"]]): args.o_tag_ratio * n_total
        }
    else:
        cluster_family_sizes = None


    print("Fitting kmeans on the semi-supervised dataset: (train and dev data unlabeled + small support labeled)\n", file=sys.stderr, flush=True)

    evaluate = make_evaluate(evaluator, decoder, transition_weights, tag_dict, cluster2tag)

    fit_start_time = time.time()
    kmeans = LDAKMeans(
        assignments_type=args.assignments,
        n_clusters=len(cluster2tag),
        init=args.centroids_init,
        device=args.device
    ).fit(
        X=train_features,
        y=train_tags,
        tag2clusters=tag2clusters,
        cluster2tag=cluster2tag,
        true_y=train_gold_tags,
        evaluate=evaluate,
        cluster_family_sizes=cluster_family_sizes,
        n_iter=args.n_iter,
        kmeans_n_iter=args.kmeans_n_iter,
        n_iter_bregman=args.n_iter_bregman,
    )

    evaluation_start_time = time.time()

    print("--------------------------------------------------------------------------", file=sys.stderr, flush=True)
    print("kmeans fit number of iterations: {}".format(kmeans.n_iter), file=sys.stderr, flush=True)
    print("kmeans fit execution time: {} (s)".format(evaluation_start_time - fit_start_time), file=sys.stderr, flush=True)
    print("--------------------------------------------------------------------------", file=sys.stderr, flush=True)
    print("evaluation on the test set... ", end="", file=sys.stderr, flush=True)
    evaluate(kmeans, test_features, test_tags)
    print("{} (s)".format(time.time() - evaluation_start_time), file=sys.stderr, flush=True)

    results = {
        "precision": evaluator.precision() * 100,
        "recall": evaluator.recall() * 100,
        "f1": evaluator.f1() * 100,
    }
    print("results: {}".format(results), file=sys.stderr, flush=True)
    print("results: {}".format(results), flush=True)
    print("--------------------------------------------------------------------------\n", file=sys.stderr, flush=True)

    for k in all_results.keys():
        all_results[k].append(results[k])

print("final results over {} support sets:".format(len(train_supports_data)), flush=True)
print("    precision: {} +- {}".format(np.mean(all_results["precision"]), np.std(all_results["precision"])), flush=True)
print("    recall   : {} +- {}".format(np.mean(all_results["recall"]), np.std(all_results["recall"])), flush=True)
print("    f1       : {} +- {}".format(np.mean(all_results["f1"]), np.std(all_results["f1"])), flush=True)
print("{} +- {}".format(np.mean(all_results["f1"]), np.std(all_results["f1"])), flush=True)
