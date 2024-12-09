import torch
import torch.nn as nn
import transformers


class NetworkForTokenClassification(nn.Module):
    def __init__(self,
        bert_model,
        num_tags,
        dropout=0,
        bert_cache="",
        bert_proxy="",
        logits_via_distance=False
    ):
        super().__init__()

        self.bert = transformers.AutoModel.from_pretrained(
            bert_model,
            proxies={"https": bert_proxy} if len(bert_proxy) > 0 else None,
            cache_dir=bert_cache if len(bert_cache) > 0 else None,
        )
        self.dropout = nn.Dropout(dropout)
        
        self.logits_via_distance = logits_via_distance
        self.output_projection = nn.Linear(self.bert.config.hidden_size, num_tags, bias=not logits_via_distance)

        self.transition_weights = nn.Parameter(torch.empty(num_tags, num_tags))
        with torch.no_grad():
            self.transition_weights.zero_()

    def get_features(self, input_tensors):
        all_input_ids = torch.nn.utils.rnn.pad_sequence(
            [sentence["input"] for sentence in input_tensors],
            batch_first=True,
            padding_value=0
        )
        all_lengths = torch.tensor(
            [len(sentence["input"]) for sentence in input_tensors],
            dtype=torch.long,
            device=all_input_ids.device
        )  # length of each sequence
        all_input_mask = torch.arange(
            all_input_ids.shape[1],
            device=all_input_ids.device
        )[None, :] < all_lengths[:, None]

        features = self.bert(
            all_input_ids,
            token_type_ids=None,
            attention_mask=all_input_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )["last_hidden_state"]

        all_features = []
        if "start_mask" in input_tensors[0]:
            all_word_start_mask = torch.nn.utils.rnn.pad_sequence(
                [sentence["start_mask"] for sentence in input_tensors],
                batch_first=True,
                padding_value=0
            )
            all_features.append(
                features.masked_select(all_word_start_mask.unsqueeze(-1))
            )

        if "end_mask" in input_tensors[0]:
            all_word_end_mask = torch.nn.utils.rnn.pad_sequence(
                [sentence["end_mask"] for sentence in input_tensors],
                batch_first=True,
                padding_value=0
            )
            all_features.append(features.masked_select(all_word_end_mask.unsqueeze(-1)))

        if len(all_features) == 1:
            features = all_features[0]
        else:
            features = (all_features[0] + all_features[1]) / 2

        features = features.reshape(-1, self.bert.config.hidden_size)

        return features

    def forward(self, input_tensors):

        # shape features: (num_words in batch, num features)
        features = self.get_features(input_tensors)

        if self.logits_via_distance:
            # shape self.output_projection.weights (output size, input size)
            logits = - ((self.output_projection.weight.unsqueeze(0) - features.unsqueeze(1)) ** 2).sum(dim=2)
        else:
            logits = self.output_projection(self.dropout(features))

        max_length = max(sentence["n_tokens"] for sentence in input_tensors)
        ret_logits = torch.zeros((len(input_tensors), max_length, logits.shape[1]), device=logits.device)
        start = 0
        for b, size in enumerate(sentence["n_tokens"] for sentence in input_tensors):
            ret_logits[b, 0:size, :] = logits[start:start + size, :]
            start += size

        return ret_logits, self.transition_weights

    def bert_load_state_dict(self, network_state_dict):
        bert_state_dict = {k.removeprefix("bert."): v for k, v in network_state_dict.items() if k.startswith("bert.")}
        self.bert.load_state_dict(bert_state_dict)
