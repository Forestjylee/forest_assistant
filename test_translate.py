import time
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
mode_name = 'liam168/trans-opus-mt-zh-en'
model = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)

start_time = time.time()
translation = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)
res = translation('帮我打开厨房的灯，然后进入帮我记录点东西。', max_length=400)
print(res)
print(f"Cost time: {time.time() - start_time}")
