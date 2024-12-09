import torch
import transformers


class BertInputBuilder:
    def __init__(self, tokenizer_name, proxy="", cache="", start_features=False, end_features=False, do_lower_case=False):
        assert start_features or end_features

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name,
            do_lower_case=do_lower_case,
            proxies={"https": proxy} if len(proxy) > 0 else None,
            cache_dir=cache if len(cache) > 0 else None
        )

        self.start_features = start_features
        self.end_feature = end_features

    def __call__(self, sentence, prev_sentence=None, next_sentence=None, context=0, start_boundary=False, boundaries=False, device="cpu"):
        bert_repr = dict()
        bert_repr["n_tokens"] = len(sentence) + (1 if boundaries or start_boundary else 0) + (1 if boundaries else 0)

        tokens = []
        word_start_mask = []
        word_end_mask = []

        tokens.append(self.tokenizer.cls_token)
        word_start_mask.append(1 if boundaries or start_boundary else 0)
        word_end_mask.append(1 if boundaries or start_boundary else 0)

        if prev_sentence is not None and context > 0:
            for word in prev_sentence[-context:]:
                word_tokens = self.tokenizer.tokenize(word)
                for _ in range(len(word_tokens)):
                    word_start_mask.append(0)
                    word_end_mask.append(0)
                tokens.extend(word_tokens)

        for word in sentence:
            word_tokens = self.tokenizer.tokenize(word)
            for _ in range(len(word_tokens)):
                word_start_mask.append(0)
                word_end_mask.append(0)
            # as tokens is extended later, overwrite start id with 1
            # (was init to zero just above)
            word_start_mask[len(tokens)] = 1
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)

        if next_sentence is not None and context > 0:
            for word in next_sentence[:context]:
                word_tokens = self.tokenizer.tokenize(word)
                for _ in range(len(word_tokens)):
                    word_start_mask.append(0)
                    word_end_mask.append(0)
                tokens.extend(word_tokens)

        tokens.append(self.tokenizer.sep_token)
        word_start_mask.append(1 if boundaries else 0)
        word_end_mask.append(1 if boundaries else 0)

        # all ids for the current sentence in the batch
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        bert_repr["input"] = torch.tensor(input_ids, device=device, dtype=torch.long)
        if self.start_features:
            bert_repr["start_mask"] = torch.tensor(word_start_mask, dtype=torch.bool, device=device)
        if self.end_feature:
            bert_repr["end_mask"] = torch.tensor(word_end_mask, dtype=torch.bool, device=device)

        return bert_repr


def build_torch_tensors(data, bert_input_builder, tag_dict, device="cpu", build_tags_tensor=False):
    for sentence in data:
        if build_tags_tensor:
            sentence["tags_tensor"] = torch.tensor(
                tag_dict.tags_to_idx(sentence["tags"]),
                dtype=torch.long,
                device=device
            )

        sentence["bert_repr"] = bert_input_builder(
            sentence["tokens"],
            boundaries=False,
            device=device
        )