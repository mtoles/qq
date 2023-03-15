from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import math

from metrics import get_metrics
from prepare_data import prepare_inputs_hp
from tqdm import tqdm
from nltk.corpus import stopwords

sw_set = set(stopwords.words("english"))


class Oracle:
    """abstract method for oracles"""

    def __init__(self):
        print("subclass this method")

    def process(self, ds, q2_masking_scheme):
        new_ds = ds.map(
            lambda x: self.forward(x, q2_masking_scheme),
            load_from_cache_file=False,
            batched=True,
            batch_size=1,
        )
        return new_ds

    def forward():
        print("subclass this method")


class T5_Gen_Oracle(Oracle):
    def __init__(
        self,
        model_name,
        eval_batch_size=1,
        # raw_val_dataset=None,
    ):
        self.model_name = model_name
        self.eval_batch_size = eval_batch_size
        self.model_name = f"google/flan-{self.model_name}"
        self.tk = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir="./.model_cache"
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir="./.model_cache"
        ).cuda()
        self.model.eval()

    def forward(self, example, q2_masking_scheme):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        with torch.no_grad():
            q2 = example[f"q2_{q2_masking_scheme}"]
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
            input_ids = prompt_encoding.input_ids.cuda().repeat(c, 1)
            input_attention_masks = (
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

            oracle_answer = corpus_strs[best_index]
            oracle_answer_is_correct = bool(best_index == 0)
            example[f"a2_{q2_masking_scheme}"] = [oracle_answer]
            example[f"a2_is_correct_{q2_masking_scheme}"] = [oracle_answer_is_correct]
            return example


class T5_Bool_Oracle(Oracle):
    def __init__(
        self,
        model_name,
        eval_batch_size=1,
        # raw_val_dataset=None,
    ):
        self.model_name = model_name
        self.eval_batch_size = eval_batch_size
        self.model_name = f"google/flan-{self.model_name}"
        self.tk = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir="./.model_cache"
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir="./.model_cache"
        ).cuda()
        self.model.eval()

    def forward(self, example, q2_masking_scheme):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        with torch.no_grad():
            q2 = example[f"q2_{q2_masking_scheme}"]
            masked_sentence = example["masked_sentence"]
            # Build the corpus
            # First answer is correct. The rest are distractor.
            corpus_strs = masked_sentence + [
                distractor
                for sublist in example["context_distractor"][0]["sentences"]
                for distractor in sublist
            ]
            input_strs = [
                f"question: {q2}\ncontext: {cs}\nprompt: Does the context answer the question, yes or no?"
                for cs in corpus_strs
            ]

            input_ids = self.tk(
                input_strs, return_tensors="pt", padding=True
            ).input_ids.cuda()

            c = len(corpus_strs)

            # copy input_ids for each possible answer
            label_strs = ["yes", "no"]
            label_encoding = self.tk(label_strs, return_tensors="pt", padding=True)
            max_answer_len = 1  # must change if label_strs is edited
            label_ids = label_encoding.input_ids[:, :-1].cuda()
            label_attention_masks = label_encoding.attention_mask[:, :-1].cuda()

            # process logits in batches
            num_batches = math.ceil(c / self.eval_batch_size)
            probs = []
            for i in range(num_batches):
                start = i * self.eval_batch_size
                end = min((i + 1) * self.eval_batch_size, c)
                # batch_logits = self.model(
                #     input_ids=input_ids[start:end], labels=label_ids
                # ).logits
                batch_logits = self.model.generate(
                    input_ids[start:end],
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                ).scores[0]
                yn_scores = batch_logits[:, label_ids.T.squeeze()].softmax(dim=1)
                # batch_prob = (
                # yn_scores.softmax(dim=1)
                # .view((end - start) * max_answer_len, -1)[
                #     torch.arange((end - start) * max_answer_len),
                #     label_ids.flatten(),
                # ]
                # .view(end - start, -1)
                # .log()
                # .mul(label_attention_masks)
                # .sum(dim=1)
                # )
                # probs.append(batch_prob)
                probs.append(yn_scores)

            probs = torch.cat(probs, dim=0)

            best_index = probs[:, 0].argmax()

            oracle_answer = corpus_strs[best_index]
            oracle_answer_is_correct = bool(best_index == 0)
            example[f"a2_{q2_masking_scheme}"] = [oracle_answer]
            example[f"a2_is_correct_{q2_masking_scheme}"] = [oracle_answer_is_correct]
            return example


class Word_Overlap_Oracle(Oracle):
    def __init__(self, eval_batch_size=1):
        self.eval_batch_size = eval_batch_size

    def forward(self, example, q2_masking_scheme):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        with torch.no_grad():
            q2 = example[f"q2_{q2_masking_scheme}"][0]  # assume batch size 1
            masked_sentence = example["masked_sentence"]
            # Build the corpus
            # First answer is correct. The rest are distractor.
            corpus_strs = masked_sentence + [
                distractor
                for sublist in example["context_distractor"][0]["sentences"]
                for distractor in sublist
            ]

            def _bow_f1(a: str, b: str):
                a = set(a.split()) - sw_set
                b = set(b.split()) - sw_set
                # drop stop words from a and b

                return 2 * len(a & b) / (len(a) + len(set(b)))

            f1_scores = [_bow_f1(q2, a) for a in corpus_strs]
            best_index = f1_scores.index(max(f1_scores))

            oracle_answer = corpus_strs[best_index]
            oracle_answer_is_correct = bool(best_index == 0)
            example[f"a2_{q2_masking_scheme}"] = [oracle_answer]
            example[f"a2_is_correct_{q2_masking_scheme}"] = [oracle_answer_is_correct]
            return example
