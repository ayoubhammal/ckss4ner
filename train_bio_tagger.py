import argparse
import sys
import io
import time
import random
import itertools

import torch
import numpy as np
import transformers
from ratio_constraints.optim import mAdamW

from ratio_constraints.network import NetworkForTokenClassification
from ratio_constraints.utils import batchify, TagDict
import ratio_constraints.repr
import ratio_constraints.io
from ratio_constraints.loss_builders import tagging_loss_builder_factory
from ratio_constraints.decoders import TaggingDecoder
from ratio_constraints.eval import NEREvaluator


cmd = argparse.ArgumentParser()

cmd.add_argument("--model", type=str, required=True)
cmd.add_argument("--train", type=str, required=True)
cmd.add_argument("--max-train-length", type=int, default=0)
cmd.add_argument("--dev", type=str, required=True)
cmd.add_argument("--test", type=str, required=True)

cmd.add_argument("--logits", choices=["proj", "dist"], required=True)
cmd.add_argument("--loss", type=str, required=True)
cmd.add_argument("--err-ratio", type=float, default=0.85)
cmd.add_argument("--err-tradeoff", type=float, default=10.)
cmd.add_argument("--err-margin", type=float, default=0.05)

cmd.add_argument("--n-epochs", type=int, default=20)
cmd.add_argument("--lr", type=float, default=1e-5)  # 1e-6
cmd.add_argument("--scheduler", type=str, default="linear")
cmd.add_argument("--minibatch-size", type=int, default=300)
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
cmd.add_argument("--clip-norm", type=float, default=1)
cmd.add_argument("--weight-decay", type=float, default=0.01)
cmd.add_argument("--warmup-ratio", type=float, default=0.1)

cmd.add_argument("--seed", type=int, default=0)

args = cmd.parse_args()

if args.seed != 0:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

train_data = ratio_constraints.io.read(args.train)
if args.max_train_length > 0:
    train_data = [sentence for sentence in train_data if len(sentence["tokens"]) <= args.max_train_length]
dev_data = ratio_constraints.io.read(args.dev)
test_data = ratio_constraints.io.read(args.test)

# check if loss is compatible with training data
has_X = any(any(t == "X" for t in sentence["tags"]) for sentence in train_data)
assert (args.loss == "supervised" and not has_X) or (args.loss != "supervised" and has_X)

tag_dict = TagDict(
    set(
        itertools.chain(*[sentence["tags"] for sentence in train_data])
    ).union({"O"})  # O may not exist in partially labeled training datasets
)

transition_mask = tag_dict.build_transition_mask(dtype=torch.float, device=args.device)
beginning_mask = tag_dict.build_beginning_mask(dtype=torch.float, device=args.device)

decoder = TaggingDecoder(transition_mask=transition_mask, beginning_mask=beginning_mask)
evaluator = NEREvaluator()

loss_builder = tagging_loss_builder_factory(
    loss=args.loss,
    o_tag_idx=tag_dict.tag_to_id["O"],
    err_ratio=args.err_ratio,
    err_tradeoff=args.err_tradeoff,
    err_margin=args.err_margin,
    bregman_ratio=None,
    bregman_eps=None,
    reduction="sum",
    transition_mask=transition_mask,
    beginning_mask=beginning_mask
)

bert_input_builder = ratio_constraints.repr.BertInputBuilder(
    tokenizer_name=args.bert_tokenizer,
    proxy=args.bert_proxy,
    cache=args.bert_cache,
    start_features=args.bert_start_features,
    end_features=args.bert_end_features,
    do_lower_case=args.bert_do_lower_case
)

ratio_constraints.repr.build_torch_tensors(
    train_data,
    bert_input_builder,
    tag_dict,
    device=args.storage_device,
    build_tags_tensor=True
)
ratio_constraints.repr.build_torch_tensors(
    dev_data,
    bert_input_builder,
    tag_dict,
    device=args.storage_device
)
ratio_constraints.repr.build_torch_tensors(
    test_data,
    bert_input_builder,
    tag_dict,
    device=args.storage_device
)

train_data = batchify(
    train_data,
    args.minibatch_size,
    lambda sentence: 1
)
dev_data = batchify(
    dev_data,
    args.eval_minibatch_size,
    lambda sentence: 1
)
test_data = batchify(
    test_data,
    args.eval_minibatch_size,
    lambda sentence: 1
)

network = NetworkForTokenClassification(
    args.bert_model,
    len(tag_dict),
    dropout=args.dropout,
    bert_cache=args.bert_cache,
    bert_proxy=args.bert_proxy,
    logits_via_distance=False if args.logits == "proj" else True
)
network.to(args.device)

param_optimizer = list(network.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']  #, 'gamma', 'beta'
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}
]
optimizer = mAdamW(
    optimizer_grouped_parameters,
    lr=args.lr
)

total_steps = len(train_data) * args.n_epochs
warmup_steps = total_steps * args.warmup_ratio

scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    num_cycles=0.5
)


def eval(data, dataset_name, ostream=sys.stderr):
    evaluator.reset()
    for batch_data in data:
        with torch.no_grad():
            bert_repr = [
                {
                    k:
                        (v.to(args.device) if k != "n_tokens" else v)
                    for k, v in t["bert_repr"].items()
                }
                for t in batch_data
            ]
            batch_weights, transition_weights = network(bert_repr)
        # we need to get out of the no_grad() env to compute prediction
        preds = decoder(batch_weights, transition_weights, [len(sentence["tokens"]) for sentence in batch_data])
        # the prediction algorithm uses autograd, so we have to put that outside torch.no_grad()
        for sentence, pred in zip(batch_data, preds):
            evaluator(sentence["tags"], tag_dict.idx_to_tags(pred))

    tmp_ostream = io.StringIO()
    evaluator.write(tmp_ostream, dataset_name)
    ostream.write(tmp_ostream.getvalue())

    return evaluator.f1(), tmp_ostream.getvalue()

best_dev_epoch = -1
best_dev_f1 = float("-inf")
best_dev_results = "NaN"
best_test_results = "NaN"
optimizer.zero_grad()
for epoch in range(args.n_epochs):
    print("START EPOCH", file=sys.stderr, flush=True)
    epoch_start = time.time()
    epoch_loss = 0
    network.train()
    random.shuffle(train_data)
    for batch_data in train_data:
        optimizer.zero_grad()

        bert_repr = [sentence["bert_repr"] for sentence in batch_data]
        bert_repr = [
            {
                k:
                    (v.to(args.device) if k != "n_tokens" else v)
                    for k, v in t.items()
            }
            for t in bert_repr
        ]
        batch_weights, transition_weights = network(bert_repr)

        batch_loss = loss_builder(
            batch_weights,
            transition_weights,
            [sentence["tags_tensor"] for sentence in batch_data]
        )
        epoch_loss += batch_loss.item()
        mean_loss = batch_loss / len(batch_data)
        mean_loss.backward()
        if args.clip_norm > 0:
            parameters_to_clip = [p for p in network.parameters() if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, args.clip_norm).item()
        optimizer.step()
        scheduler.step()

    epoch_loss = epoch_loss / sum(len(b) for b in train_data)
    epoch_end = time.time()

    # dev eval
    network.eval()

    print("Epoch %i" % epoch, file=sys.stderr)
    print("########", file=sys.stderr)
    print("loss: %.6f" % epoch_loss, file=sys.stderr)
    print("train timing: %.2f" % ((epoch_end-epoch_start) / 60), file=sys.stderr, flush=True)

    eval_start = time.time()

    dev_f1, dev_results = eval(dev_data, "dev")
    print(file=sys.stderr, flush=True)

    _, test_results = eval(test_data, "test")
    print(file=sys.stderr, flush=True)

    eval_end = time.time()
    print("eval timing: %.2f" % ((eval_end-eval_start) / 60), file=sys.stderr)
    print(file=sys.stderr, flush=True)

    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_dev_results = dev_results
        best_test_results = test_results
        torch.save(
        {
                "args": args,
                "tag_dict": tag_dict,
                "model_state_dict": network.state_dict(),
            },
            args.model
        )

print("Final results on dev:\n")
print(best_dev_results)
print()
print("Final results on test:\n")
print(best_test_results)
