import torch


# https://github.com/pytorch/pytorch/issues/31829#issuecomment-581257543
def logsumexp(x, dim):
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))


def check_tagging_tensors(tag_weights, transition_weights, num_words):
    assert len(tag_weights.shape) == 3
    assert len(transition_weights.shape) == 2
    assert transition_weights.shape[0] == tag_weights.shape[2]
    assert transition_weights.shape[0] == transition_weights.shape[1]
    assert num_words == tag_weights.shape[1]


# Forward algorithm for sequence tagging
# can be used to implement loss functions
def tagging_log_partition(tag_weights, transition_weights, lengths):
    num_words = max(lengths)
    check_tagging_tensors(tag_weights, transition_weights, num_words)

    chart = tag_weights[:, 0]
    # need to transpose
    transition_weights = transition_weights.t().unsqueeze(0)
    goal_scores = torch.empty(
        tag_weights.shape[0],
        dtype=tag_weights.dtype,
        device=tag_weights.device
    )
    for idx, l in enumerate(lengths):
        if l == 1:
            goal_scores[idx] = logsumexp(chart[idx], dim=0)
    for i in range(1, num_words):
        # t[next_tag, prev_tag]
        t = chart.unsqueeze(1) + tag_weights[:, i].unsqueeze(2) + transition_weights
        chart = logsumexp(t, dim=2)
        for idx, l in enumerate(lengths):
            if i == l - 1:
                goal_scores[idx] = logsumexp(chart[idx], dim=0)

    return goal_scores

def tagging_log_partition_bregman(tag_weights, transition_weights, lengths, ratio, max_iter, eps):
    W, _ = tagging_log_marginals(tag_weights, transition_weights, lengths)
    W = W.view(-1, W.shape[2])
    ratio = ratio.clone() * tag_weights.shape[0] * tag_weights.shape[1] # ratio are percentage so we multiply by number of words and number of sentences

    with torch.no_grad():
        # Mask is already applied on tag_weights for labels
        W_proj = -W.clone() / eps
        # Put 'inf' to '-inf'
        W_proj[W_proj == float('inf')] = float('-inf')
        # Transform the log_marginals shape from B,N,C to B*N,C
        # W = W.view(-1, W.shape[2])
        for _ in range(max_iter):
            W_proj = torch.log_softmax(W_proj, dim=1).nan_to_num(float('-inf'))
            W_proj = torch.log_softmax(W_proj, dim=0) + torch.log(ratio)
        P = W_proj.exp()

    # B*N
    log_partition_bregman = torch.nan_to_num(P*W).sum(dim=1) - torch.nan_to_num(P*P.log()).sum(dim=1)
    # Transform from B*N to B N
    log_partition_bregman = log_partition_bregman.view(-1, tag_weights.shape[1])

    # Return the sum over all the words to get the log partition for each
    # sentence
    return log_partition_bregman.sum(dim=1)

# Forward-backward algorithm for sequence tagging
# the function returns tag marginals,
# this is useful for iterative Bregman projection
# as it doesn't depend on the computation graph to compute marginals.
# It is probably faster to use than torch_struct.
# It can also be used for the ERR loss.
def tagging_log_marginals(tag_weights, transition_weights, lengths):
    num_words = max(lengths)
    check_tagging_tensors(tag_weights, transition_weights, num_words)

    # Start with backward
    # (so we transpose the transition matrix only once)

    transition_weights = transition_weights.unsqueeze(0)
    backward_chart = torch.empty_like(tag_weights)
    backward_chart.fill_(float("-inf"))
    for idx, l in enumerate(lengths):
        if l == num_words:
            backward_chart[idx, num_words - 1] = 0
    for i in reversed(range(num_words - 1)):
        # t[next_tag, prev_tag]
        t = backward_chart[:, i + 1].unsqueeze(1) + tag_weights[:, i + 1].unsqueeze(1) + transition_weights
        backward_chart[:, i] = logsumexp(t, dim=2)
        for idx, l in enumerate(lengths):
            if i == l - 1:
                backward_chart[idx, i] = 0

    # Now forward

    # need to transpose
    transition_weights = transition_weights.transpose(-1, -2)
    log_z = torch.empty(
        tag_weights.shape[0],
        dtype=tag_weights.dtype,
        device=tag_weights.device
    )

    forward_chart = torch.empty_like(tag_weights)
    forward_chart[:, 0] = tag_weights[:, 0]
    for idx, l in enumerate(lengths):
        if l == 1:
            log_z[idx] = logsumexp(forward_chart[idx, 0], dim=0)

    for i in range(1, num_words):
        # t[next_tag, prev_tag]
        t = forward_chart[:, i - 1].unsqueeze(1) + tag_weights[:, i].unsqueeze(2) + transition_weights
        forward_chart[:, i] = logsumexp(t, dim=2)
        for idx, l in enumerate(lengths):
            if i == l - 1:
                log_z[idx] = logsumexp(forward_chart[idx, i], dim=0)

    # Combine results

    log_marginals = forward_chart + backward_chart - log_z.reshape(-1, 1, 1)

    return log_marginals, log_z

# Implementation of the Viterbi algorithm
#
# Warning: this uses the autograd mechanism!
# so it cannot be embedded in a torch.no_grad() env
def tagging_argmax(tag_weights, transition_weights, lengths):
    num_words = max(lengths)
    check_tagging_tensors(tag_weights, transition_weights, num_words)

    tag_weights = tag_weights.clone().detach().requires_grad_(True)
    transition_weights = transition_weights.clone().detach().requires_grad_(False)

    chart = tag_weights[:, 0]
    # need to transpose
    transition_weights = transition_weights.t().unsqueeze(0)
    goal_scores = torch.empty(
        tag_weights.shape[0],
        dtype=tag_weights.dtype,
        device=tag_weights.device
    )
    for idx, l in enumerate(lengths):
        if l == 1:
            goal_scores[idx] = torch.max(chart[idx], dim=0).values
    for i in range(1, num_words):
        # t[next_tag, prev_tag]
        t = chart.unsqueeze(1) + tag_weights[:, i].unsqueeze(2) + transition_weights
        chart = torch.max(t, dim=2).values
        for idx, l in enumerate(lengths):
            if i == l - 1:
                goal_scores[idx] = torch.max(chart[idx], dim=0).values

    goal_scores.sum().backward()

    return goal_scores, tag_weights.grad.detach().argmax(axis=2)
