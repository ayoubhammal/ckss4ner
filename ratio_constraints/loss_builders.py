import torch
from torch import nn
from ratio_constraints.tagging_algorithms import (tagging_log_partition, tagging_log_marginals,
                                                  tagging_log_partition_bregman)


def check_reduction(reduction):
    assert reduction in ["sum", "mean", "none"]


def apply_reduction(loss, reduction):
    assert reduction in ["sum", "mean", "none"]

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        return loss


# NLL loss for a linear chain CRF
# note that the computation of logits is not really parallelized,
# but I think we don't care, it should be fast enough anyway.
class TaggingSupervisedCRFLossBuilder(nn.Module):
    def __init__(self, reduction="none", transition_mask=None, beginning_mask=None):
        super().__init__()
        check_reduction(reduction)
        self.reduction = reduction
        self.transition_mask = transition_mask
        self.beginning_mask = beginning_mask

    def __call__(self, tag_weights, transition_weights, gold):
        lengths = [len(s) for s in gold]
        if self.transition_mask is not None:
            transition_weights = transition_weights + self.transition_mask
        if self.beginning_mask is not None:
            tag_weights = tag_weights.clone()
            tag_weights[:, 0] += self.beginning_mask.unsqueeze(0)
        log_partition = tagging_log_partition(tag_weights, transition_weights, lengths)

        logits = (
            # tag scores
            torch.cat([
                torch.gather(tag_weights[i], -1, gold[i].unsqueeze(-1)).squeeze(-1).sum(dim=0, keepdim=True)
                for i in range(len(gold))
            ])
            +
            # transition scores
            torch.cat([
                transition_weights[gold[i][:-1], gold[i][1:]].sum(dim=0, keepdim=True)
                for i in range(len(gold))
            ])
        )

        loss = -logits + log_partition

        return apply_reduction(loss, self.reduction)


# Partially supervised loss for a linear chain CRF
# it simply mask tag weights to compute the clamped log-partition
#
# Note that if a fully supervised example is passed,
# it will still use the forward algorithm to compute the clamped log-partition.
# This should still be ok in practice...
class TaggingPartialCRFLossBuilder(nn.Module):
    def __init__(self, reduction="none", transition_mask=None, beginning_mask=None):
        super().__init__()
        check_reduction(reduction)
        self.reduction = reduction
        self.transition_mask = transition_mask
        self.beginning_mask = beginning_mask

    def __call__(self, tag_weights, transition_weights, gold, return_clamped_weights=False):
        lengths = [len(s) for s in gold]
        if self.transition_mask is not None:
            transition_weights = transition_weights + self.transition_mask
        if self.beginning_mask is not None:
            tag_weights = tag_weights.clone()
            tag_weights[:, 0] += self.beginning_mask.unsqueeze(0)
        ar = torch.arange(tag_weights.shape[1], device=tag_weights.device)

        mask = torch.empty_like(tag_weights)
        mask.fill_(float("-inf"))

        for i, g in enumerate(gold):
            mask[i, ar[:len(g)], g] = 0
            mask[i][:len(g)][g < 0] = 0

        clamped_tag_weights = tag_weights + mask

        loss = (
            tagging_log_partition(tag_weights, transition_weights, lengths)
            -
            tagging_log_partition(clamped_tag_weights, transition_weights, lengths)
        )

        loss = apply_reduction(loss, self.reduction)

        if return_clamped_weights:
            return loss, clamped_tag_weights
        else:
            return loss


def tagging_loss_builder_factory(loss, o_tag_idx, err_ratio, err_tradeoff, err_margin, bregman_eps, bregman_ratio, reduction='none', transition_mask=None, beginning_mask=None):
    if loss == "supervised":
        return TaggingSupervisedCRFLossBuilder(reduction=reduction, transition_mask=transition_mask, beginning_mask=beginning_mask)
    else:
        raise RuntimeError("Unknown loss : %s" % loss)
