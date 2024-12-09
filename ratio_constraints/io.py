

def read(input_file, unk_tag="X"):
    data = list()
    with open(input_file) as instream:
        for sentence in instream:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue

            tags = next(instream).strip()

            sentence = sentence.split()
            tags = tags.split()

            assert len(sentence) == len(tags)
            assert all(t in ("O", unk_tag) or t[:2] in ("B-", "I-") for t in tags)

            data.append({
                "tokens": sentence,
                "tags": tags
            })

    return data
