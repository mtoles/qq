"""
Convert a json dataset created with `make_jeopardy_ds.py` or `make_jeopardy_gpt_ds.py` to a dataset of the exact same form as Tatsu lab's original `alpaca_data.json`.

Output dataset will be in a jsonl format with each line containing the keys "instruction", "input", "output".
"""

import pandas as pd
import json


def fit_template(q1, context):
    prompt_id = "p1"

    input_template = "Question:\n{q1}\nContext:\n{context}"
    instruction_prefix = (
        "What question can you ask to help you answer the final question:"  # p3
    )
    inpt = input_template.format(context=context, q1=q1)
    instruction = f"{instruction_prefix}\n\n{inpt}"
    # prompt = self.alpaca_template.format(instruction=instruction, input=inpt)

    return instruction


input_ds_path = "data/jeopardy/jeopardy_gpt_1000_train.jsonl"
output_ds_path = input_ds_path.split(".")[-2] + "_tatsu.jsonl"
# output_ds_path = "data/jeopardy/jeopardy_4000_train_active_filtered_tatsu.jsonl"


df = pd.read_json(input_ds_path, lines=True)
# lambda x: {"instruction": alpaca.fit_template(x["q1"], x["fc_masked"])}

df["instruction"] = df.apply(lambda x: fit_template(x["q1"], x["fc_masked"]), axis=1)
df["output"] = df["jeopardy_q"]
df["input"] = ""

# write instructions, output, and input to a jsonl file as a list of dicts
output_list = df[["instruction", "input", "output"]].to_dict(orient="records")
with open(output_ds_path, "w") as f:
    f.write("[")
    f.write(",\n".join([json.dumps(line) for line in output_list]))
    f.write("]")
