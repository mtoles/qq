from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import math

from metrics import get_metrics
from prepare_data import prepare_inputs_hp
from tqdm import tqdm


class Dummy_Oracle:
    def __init__(self, corpus):
        self.corpus = corpus.split(". ")

    def consult(self, query):
        # return the first sentence of the corpus
        return self.corpus[0]


class T5_Oracle:
    def __init__(
        self,
        eval_batch_size=1,
        # raw_val_dataset=None,
        model_name=None,
    ):
        self.eval_batch_size = eval_batch_size
        model_name = f"google/flan-{model_name}"
        self.tk = AutoTokenizer.from_pretrained(model_name, cache_dir="./.model_cache")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir="./.model_cache"
        ).cuda()
        self.model.eval()

    def process(self, ds, q2_col):
        new_ds = ds.map(
            lambda x: self.forward(x, q2_col),
            load_from_cache_file=False,
            batched=True,
            batch_size=1,
        )
        return new_ds

    def forward(self, example, q2_col):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        with torch.no_grad():
            q2 = example[q2_col]
            masked_sentence = example["masked_sentence"]
            # Build the corpus
            # First answer is correct. The rest are distractor.
            corpus_strs = masked_sentence + [
                distractor
                for sublist in example["context_distractor"][0]["sentences"]
                for distractor in sublist
            ]
            corpus_ids = self.tk(corpus_strs, return_tensors="pt", padding=True)[
                "input_ids"
            ].cuda()
            max_answer_len = max([len(x) for x in corpus_ids])
            c = len(corpus_strs)

            prompt_encoding = self.tk(q2, return_tensors="pt", padding=True)
            input_ids, input_attention_masks = (
                prompt_encoding.input_ids.cuda().repeat(c, 1),
                prompt_encoding.attention_mask.cuda().repeat(c, 1),
            )

            # copy input_ids for each possible answer
            label_encoding = self.tk(corpus_strs, return_tensors="pt", padding=True)
            label_ids, label_attention_masks = (
                label_encoding.input_ids.cuda(),
                label_encoding.attention_mask.cuda(),
            )

            # process logits in batches
            num_batches = math.ceil(c / self.eval_batch_size)
            probs = []
            for i in range(num_batches):
                start = i * self.eval_batch_size
                end = min((i + 1) * self.eval_batch_size, c)
                batch_logits = self.model(
                    input_ids=input_ids[start:end], labels=label_ids[start:end]
                ).logits
                batch_prob = (
                    batch_logits.softmax(dim=2)
                    .view((end - start) * max_answer_len, -1)[
                        torch.arange((end - start) * max_answer_len),
                        label_ids[start:end].flatten(),
                    ]
                    .view(end - start, -1)
                    .log()
                    .mul(label_attention_masks[start:end])
                    .sum(dim=1)
                )
                probs.append(batch_prob)

            probs = torch.cat(probs, dim=0)


            best_index = probs.argmax()

            # TODO: find the max probability answer in probs
            oracle_answer = corpus_strs[best_index]
            oracle_answer_is_correct = bool(best_index == 0)
            example["a2"] = [oracle_answer]
            example["a2_is_correct"] = [oracle_answer_is_correct]
            return example
