from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import transformers as t
from datetime import datetime
from datasets import load_from_disk

ds = load_from_disk("data2/hotpot_train_100_randomsentence")
config = t.AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b")
config.max_position_embeddings = 3072
# config.max_position_embedding = 102
model = (
    GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)
    .half()
    .cuda()
)
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")


start = datetime.now()
for i in range(10):
    prompt = " ".join(
        [
            ds["fc_None"][i],
            ds["question"][i],
            "Answer in as few words as possible. Answer: ",
        ]
    )
    print("*****" + prompt + "*****")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    if len(input_ids) > 2047:
        continue

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_new_tokens=20,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)
    print
delta = datetime.now() - start
print(delta)
print
