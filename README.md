# Constrained K-means with Subspace Selection (CKSS) for weakly-supervised named entity recognition

This is the official repository for the paper: [Few-Shot Domain Adaptation for Named-Entity Recognition via Joint Constrained k-Means and Subspace Selection](https://arxiv.org/abs/2412.00426v1#)

## Fine-tuning on the source dataset

For pre-traiing the model on the source dataset, the `train_io_tagger.py` script can be used as shown below, all the script options can be found in the script header.
We illustrate her an example with the IO tags as it is the main tagging schema discussed in the paper, but the BIO tagging schema can be used similarly with its corresponding script versions.

```sh
python train_io_tagger.py \
    --model [model checkpoint path] \
    --train \
    --dev \
    --test \
    --logits["proj", "dist"] \
    --n-epochs 5 \
    --lr 1e-5 \
    --minibatch-size 32 \
    --eval-minibatch-size 32 \
    --bert-tokenizer bert-base-cased \
    --bert-model bert-base-cased \
    --bert-start-features \
    --bert-end-features
```

## Training and evaluation of the few-shot model

For training the few-shot model, the script needs a train and dev sets as well as the support data, the test set is used at the end for evaluation.

```sh
python evaluate_kmeans_io.py \
    --train [train split file] \
    --pretrain-train [train split of the source dataset] \
    --dev [dev split file] \
    --test [test split file] \
    --supports [support data file] \
    --target-set-tags-file [a file containing the target tag set, one tag per line] \
    --n-iter-bregman 10 \
    --n-iter 10 \
    --kmeans-n-iter 1 \
    --o-tag-ratio 0.95 \
    --n-o-clusters 10 \
    --n-i-clusters 1 \
    --assignments "hard" \
    --tau 0.05 \
    --centroids-init hierarchical-ward \
    --device cuda \
    --storage-device cuda \
    --bert-model bert-base-cased \
    --bert-tokenizer bert-base-cased \
    --model [model checkpoint path] \
    --bert-start-features \
    --bert-end-features
```

## Data format

The scripts expect each data split to be in a single file with a format illustrated in the example data under the `data/data_format_example.txt` folder.

## Tagset extension experiment supports

For copyright concerns, we only share the ids of the support sets data of the tagset extension experiment on Ontonotes5.
Those can be found in `data/support_data_ids.json`.
