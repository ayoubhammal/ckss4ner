from ratio_constraints.tagging_algorithms import tagging_argmax


class TaggingDecoder:
    def __init__(self, transition_mask=None, beginning_mask=None):
        self.transition_mask = transition_mask
        self.beginning_mask = beginning_mask

    def __call__(self, tag_weights, transition_weights, lengths):
        if self.transition_mask is not None:
            transition_weights = transition_weights + self.transition_mask
        if self.beginning_mask is not None:
            tag_weights = tag_weights.clone()
            tag_weights[:, 0] += self.beginning_mask.unsqueeze(0)

        _, tags = tagging_argmax(tag_weights, transition_weights, lengths)

        return [tags[i, :l] for i, l in enumerate(lengths)]

