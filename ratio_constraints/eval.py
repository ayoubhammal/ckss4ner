from ratio_constraints.utils import bio_to_mentions


# will convert BIO tags to mentions on the fly
class NEREvaluator(object):
    def __init__(self):
        self.reset()

    def __call__(self, gold_tags, pred_tags):
        gold_mentions = bio_to_mentions(gold_tags)
        pred_mentions = bio_to_mentions(pred_tags)

        self.n_correct += len(gold_mentions.intersection(pred_mentions))
        self.n_gold += len(gold_mentions)
        self.n_pred += len(pred_mentions)

    def reset(self):
        self.n_correct = 0
        self.n_gold = 0
        self.n_pred = 0

    def precision(self):
        if self.n_pred == 0:
            return 0
        else:
            return self.n_correct / self.n_pred

    def recall(self):
        if self.n_gold == 0:
            return 0
        else:
            return self.n_correct / self.n_gold

    def f1(self):
        prec = self.precision()
        rec = self.recall()
        if prec + rec == 0:
            return 0
        else:
            return 2 * prec * rec / (prec + rec)

    def write(self, ostream, dataset_name):
        ostream.write("Eval on %s:\tprec=%.2f\trec=%.2f\tf1=%.2f" % (
            dataset_name,
            100 * self.precision(),
            100 * self.recall(),
            100 * self.f1()
        ))
