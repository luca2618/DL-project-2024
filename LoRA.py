import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
import json
import sys


torch_device = 'cuda:0'

# List of available models
models = { # name : path (model, dataset)
    "Mistral-7B-v0.1" : "mistralai/Mistral-7B-v0.1",
    "Mistral-7B-Instruct-v0.1" : "mistralai/Mistral-7B-Instruct-v0.1",
    "BioMistral-7B" : "BioMistral/BioMistral-7B",
    "MetaMath-Mistral-7B" : "meta-math/MetaMath-Mistral-7B"
}
datasets = {
    "Math" : "meta-math/MetaMathQA",
    "Bio" : "qiaojin/PubMedQA", # Dataset doesn't exist (Find another one)
}
labels = {"Math" : ["query", "reponse"],
          "Bio" : ["question", "long_answer"]
          }

# 1. Choose and pre-download the model and tokenizer
#model_name = "Mistral-7B-Instruct-v0.1" # Choose a model from the list above
#dataset_name = "Math" # Choose a dataset from the list above



model_name= sys.argv[1]
dataset_name = sys.argv[2]

question_label = labels[dataset_name][0]
answer_label = labels[dataset_name][1]

print("selected:")
print("model:"+model_name)
print("dataset:"+dataset_name)

tokenizer = AutoTokenizer.from_pretrained(models[model_name], use_auth_token="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", cache_dir='./cache', device_map=torch_device)
model = AutoModelForCausalLM.from_pretrained(models[model_name], use_auth_token="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", cache_dir='./cache', torch_dtype=torch.float16, device_map=torch_device)
tokenizer.pad_token = tokenizer.eos_token

# 2. Define the LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers   # Check if this is correct
    lora_dropout=0.1,  # Dropout probability
    bias="none"  # Don't train biases
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

# 3. Prepare the dataset
if dataset_name == "Bio":
    ds = load_dataset(datasets[dataset_name], "pqa_artificial", cache_dir='./cache')
else:
    ds = load_dataset(datasets[dataset_name], cache_dir='./cache')

ds = ds['train'].to_pandas()[:120000]

# Split the dataset
split = 0.9
split_idx = int(len(ds) * split)
train_data_raw = ds[:split_idx]
eval_data_raw = ds[split_idx:]

# Preprocess the dataset
train_data = []
eval_data = []
for i in range(len(train_data_raw)):
    train_data.append({"prompt": train_data_raw.iloc[i][question_label], "answer": train_data_raw.iloc[i][answer_label]})
for i in range(len(eval_data_raw)):
    eval_data.append({"prompt": eval_data_raw.iloc[i][question_label], "answer": eval_data_raw.iloc[i][answer_label]})

def preprocess_function(example):
    prompt = example["prompt"]
    answer = example["answer"]
    tokenized = tokenizer(
        prompt,
        answer,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

train_dataset = Dataset.from_list(train_data).map(preprocess_function, batched=True)
eval_dataset = Dataset.from_list(eval_data).map(preprocess_function, batched=True)

# 4. Training configuration
training_args = TrainingArguments(
    output_dir=f"lora_model-{model_name}_dataset-{dataset_name}_v1",
    evaluation_strategy="steps", # Evaluate every 500 steps
    save_strategy="steps", # Save every 500 steps
    save_steps=500,
    per_device_train_batch_size=2, # Batch size per GPU
    gradient_accumulation_steps=4, # Accumulate gradients
    num_train_epochs=1.5,
    learning_rate=2e-4,
    fp16=True, # Use mixed precision
    logging_strategy="steps", # Log every 100 steps
    logging_dir="./logs", # Logs
    logging_steps=100, # Log every 100 steps
    save_total_limit=2, # Save only the last 2 checkpoints
    report_to="none", # Don't report to Hugging Face
)

# 5. Trainer setup
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    tokenizer=tokenizer
)

# 6. Fine-tune the model
trainer.train()

# File path where you want to save the log
log_file_path = f"./log/{model_name}_{dataset_name}_log_history.json"

# Save log history to a file
with open(log_file_path, "w") as log_file:
    json.dump(trainer.state.log_history, log_file, indent=4)

# 7. Save the fine-tuned LoRA model
model.save_pretrained(f"./trained_models/lora_model-{model_name}_dataset-{dataset_name}_v1")
tokenizer.save_pretrained(f"./trained_models/lora_model-{model_name}_dataset-{dataset_name}_v1")