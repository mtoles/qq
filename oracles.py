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
        eval_batch_size=4,
        # raw_val_dataset=None,
        model_name=None,
    ):
        self.eval_batch_size = eval_batch_size
        model_name = f"google/flan-{model_name}"
        self.tk = AutoTokenizer.from_pretrained(model_name, cache_dir="./.model_cache")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir="./.model_cache"
        ).cuda()

    def process(self, dataset, secondary_question_col):
        new_dataset = dataset.map(
            lambda x: self.forward(x, secondary_question_col),
            load_from_cache_file=False,
            batched=True,
            batch_size=1,
        )
        return new_dataset

    def forward(self, example, secondary_question_col):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        secondary_question = example[secondary_question_col]
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

        # question_input_ids = self.tk.encode(
        #     example[question_col], return_tensors="pt"
        # ).cuda()

        # input_strs = ["\n\n".join([example[question_col], x]) for x in self.corpus_strs]
        prompt_encoding = self.tk(secondary_question, return_tensors="pt", padding=True)
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
        for i in range(num_batches):
            start = i * self.eval_batch_size
            end = min((i + 1) * self.eval_batch_size, c)
            batch_logits = self.model(
                input_ids=input_ids[start:end], labels=label_ids[start:end]
            ).logits
            if i == 0:
                logits = batch_logits
            else:
                logits = torch.cat([logits, batch_logits], dim=0)

        # since all logits are negative, does this not make short answers artificially more likely?
        probs = (
            logits.softmax(dim=2)
            .view(c * max_answer_len, -1)[
                torch.arange(c * max_answer_len), label_ids.flatten()
            ]
            .view(c, -1)
            .log()
            .mul(label_attention_masks)
            .sum(dim=1)
        )

        best_index = probs.argmax()

        # TODO: find the max probability answer in probs
        oracle_answer = corpus_strs[best_index]
        oracle_answer_is_correct = bool(best_index == 0)
        example["oracle_answer"] = [oracle_answer]
        example["oracle_answer_is_correct"] = [oracle_answer_is_correct]
        return example

    def prepare_data(self, masking_scheme):
        masking_str = f"fc_{masking_scheme}"
        # replace [SEP] with :
        # def _replace_sep(x):
        #     assert (
        #         len(x[masking_str].split("[SEP]")) == 2
        #     ), "masking_str must contain exactly one [SEP]"
        #     x[masking_str] = x[masking_str].replace(
        #         "[SEP]", "\n\n"
        #     )  # TODO: fix. this is getting stripped out by the split/join
        #     return x

        def _add_prompt(x):
            x[masking_str] = x[masking_str] + " Answer in as few words as possible: "
            return x

        # Prepare the dataset
        # self.prepped_val_dataset = self.raw_val_dataset.map(lambda x: _replace_sep(x))
        self.prepped_val_dataset = self.prepped_val_dataset.map(
            lambda x: _add_prompt(x),
            load_from_cache_file=False,
        )
        self.prepped_val_dataset = self.prepped_val_dataset.map(
            lambda x: prepare_inputs_hp(
                x,
                tk=self.tk,
                max_length=2048,
                masking_scheme=masking_scheme,
            ),
            load_from_cache_file=False,
        )

    def evaluate(self):
        with torch.no_grad():
            str_pred = []
            str_gt = []
            cls_pred = []
            cls_gt = []
            input_ids = []

            for i, x in enumerate(tqdm(self.prepped_val_dataset)):
                generation = self.forward(input_ids=x["input_ids"])
                generation_str = self.tk.decode(generation[0])
                str_pred.append(
                    self.tk.batch_decode(generation, skip_special_tokens=True)[0]
                )
                str_gt.append(x["answer"])
                cls_pred.append(None)
                cls_gt.append(None)
                input_ids.append(x["input_ids"])
                # print("")
                # print(self.tk.decode(input_ids[-1]))
                # print(str_pred[-1])
                # print(str_gt[-1])
                # print("")

            metrics = get_metrics(
                str_pred,
                str_gt,
                cls_pred,
                cls_gt,
                input_ids,
                self.tk,
                None,
            )
        return metrics
