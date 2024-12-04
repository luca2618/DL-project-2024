
import sys
import os
os.environ['HF_HOME'] = '/dtu/blackhole/00/167776/cache/'

model_path= sys.argv[1]

from transformers import AutoModelForCausalLM, AutoTokenizer

torch_device = 'cuda:0'

model = AutoModelForCausalLM.from_pretrained(model_path, device_map=torch_device)
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=torch_device)

while True:
    input_text = input()
    inputs = tokenizer(input_text, return_tensors="pt").to(torch_device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

