import time
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
prompt="Issue:I am unable to login to my computer. content: Speeding up a slow computer: Run fewer programs at the same time Restart your computer Remove viruses and m>
input_ids = tokenizer.encode(prompt, return_tensors="pt")
st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
output = model.generate(input_ids,max_new_tokens=4000)
end = time.time()
output_str = tokenizer.decode(output[0], skip_special_tokens=True)
print(f'Inference time: {end-st} s')
print('-'*20, 'Output', '-'*20)
print(output_str)