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
            batch_size=1,  # batching happens internally
        )
        return new_ds

    def forward():
        print("subclass this method")


class T5_Bool_Oracle(Oracle):
    def __init__(
        self,
        model_size,
        batch_size,
        # raw_val_dataset=None,
    ):
        self.batch_size = batch_size
        self.model_size = model_size
        self.model_name = f"google/flan-t5-{self.model_size}"
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
            q2 = example[f"q2_{q2_masking_scheme}"][0]
            masked_sentence = example["masked_sentence"][0]
            masked_sentence_title = example["masked_sentence_title"][0]
            # Build the corpus
            # First answer is correct. The rest are distractor.
            # corpus_strs = masked_sentence + [
            #     distractor
            #     for sublist in example["context_distractor"][0]["sentences"]
            #     for distractor in sublist
            # ]
            cs_template = "%s: %s"
            corpus_strs = [
                cs_template % (masked_sentence_title, masked_sentence)
            ]  # make sure a2 is always at index 0
            # add distractors
            for i, sublist in enumerate(example["context_distractor"][0]["sentences"]):
                for distractor in sublist:
                    title = example["context_None"][0]["title"][i]
                    # corpus_str = cs_template % (title, distractor)
                    corpus_str = distractor
                    corpus_strs.append(corpus_str)
            # add supporting facts
            for i, sublist in enumerate(example["context_supporting"][0]["sentences"]):
                for supporting in sublist:
                    title = example["context_None"][0]["title"][i]
                    # corpus_str = cs_template % (title, supporting)
                    corpus_str = supporting
                    corpus_strs.append(corpus_str)
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
            label_attention_masks = label_encoding.attention_mask[:, :-1].cuda() # different from bloom

            # process logits in batches
            num_batches = math.ceil(c / self.batch_size)
            probs = []
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, c)
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
                probs.append(yn_scores)

            probs = torch.cat(probs, dim=0)

            best_index = probs[:, 0].argmax()
            best_prob = probs[:, 0].max()
            if best_prob > 0.5:
                oracle_answer = corpus_strs[best_index]
                oracle_answer_is_correct = bool(best_index == 0)
            else:  # no answer is good enough
                oracle_answer = ""
                oracle_answer_is_correct = False
            example[f"a2_{q2_masking_scheme}"] = [oracle_answer]
            example[f"a2_is_correct_{q2_masking_scheme}"] = [oracle_answer_is_correct]
            return example


class Bloom_Bool_Oracle(Oracle):
    def __init__(
        self,
        model_size,
        batch_size,
        # raw_val_dataset=None,
    ):
        self.batch_size = batch_size
        self.model_size = model_size
        self.model_name = f"bigscience/bloomz-{self.model_size}"
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
            q2 = example[f"q2_{q2_masking_scheme}"][0]
            masked_sentence = example["masked_sentence"][0]
            masked_sentence_title = example["masked_sentence_title"][0]
            # Build the corpus
            # First answer is correct. The rest are distractor.
            # corpus_strs = masked_sentence + [
            #     distractor
            #     for sublist in example["context_distractor"][0]["sentences"]
            #     for distractor in sublist
            # ]
            cs_template = "%s: %s"
            corpus_strs = [
                cs_template % (masked_sentence_title, masked_sentence)
            ]  # make sure a2 is always at index 0
            # add distractors
            for i, sublist in enumerate(example["context_distractor"][0]["sentences"]):
                for distractor in sublist:
                    title = example["context_None"][0]["title"][i]
                    # corpus_str = cs_template % (title, distractor)
                    corpus_str = distractor
                    corpus_strs.append(corpus_str)
            # add supporting facts
            for i, sublist in enumerate(example["context_supporting"][0]["sentences"]):
                for supporting in sublist:
                    title = example["context_None"][0]["title"][i]
                    # corpus_str = cs_template % (title, supporting)
                    corpus_str = supporting
                    corpus_strs.append(corpus_str)
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
            # label_ids = label_encoding.input_ids[:, :-1].cuda() # different from flan-t5
            label_ids = label_encoding.input_ids.cuda()
            label_attention_masks = label_encoding.attention_mask[:, :-1].cuda()

            # process logits in batches
            num_batches = math.ceil(c / self.batch_size)
            probs = []
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, c)
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
                probs.append(yn_scores)

            probs = torch.cat(probs, dim=0)

            best_index = probs[:, 0].argmax()
            best_prob = probs[:, 0].max()
            if best_prob > 0.5:
                oracle_answer = corpus_strs[best_index]
                oracle_answer_is_correct = bool(best_index == 0)
            else:  # no answer is good enough
                oracle_answer = ""
                oracle_answer_is_correct = False
            example[f"a2_{q2_masking_scheme}"] = [oracle_answer]
            example[f"a2_is_correct_{q2_masking_scheme}"] = [oracle_answer_is_correct]
            return example
