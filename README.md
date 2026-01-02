Fine-Tuning LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations
Project Overview

This project fine-tunes LLaMA 3.1-8B-Instruct to generate empathetic responses in Bengali.
We use LoRA for efficient fine-tuning on limited GPU resources, keeping full sequence length for long conversations.

Features

Preprocesses raw Bengali conversation CSVs into LLM instruction format.

Fine-tunes LLaMA using LoRA adapters (only a small number of parameters are updated).

Evaluation metrics:

Perplexity (validation set)

BLEU & ROUGE (test set)

Sample human evaluation for empathy.

Logging:

LLAMAExperiments.jsonl → train/val loss, LoRA config, metrics

GeneratedResponses.jsonl → input text, model responses, timestamps

Requirements

Kaggle free GPU (16GB+ recommended)

Python 3.10+

Libraries: transformers, accelerate, peft, datasets, evaluate, sentence-transformers, bitsandbytes

Hugging Face account with access to meta-llama/Meta-Llama-3.1-8B-Instruct

Setup

Install dependencies

!pip install -q --upgrade bitsandbytes==0.41.0
!pip install -q --upgrade transformers accelerate peft datasets evaluate sentence-transformers


Set your Hugging Face token

HF_TOKEN = "your_hf_token_here"


Make sure your token has read access to gated repos.

Usage
1. Preprocess Dataset
processor = DatasetProcessor(data_path="BengaliEmpatheticConversations.csv")
processor.load_raw_dataset()
dataset = processor.prepare_instruction_dataset(tokenizer=tokenizer)

2. Load Model and Tokenizer (8-bit for Kaggle)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    quantization_config=quant_config,
    llm_int8_enable_fp32_cpu_offload=True,
    token=HF_TOKEN
)

3. Apply LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

4. Train Model
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./llama-bengali-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=20,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=200,
    optim="adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=8192,
    args=training_args,
)
trainer.train()

5. Evaluate Model
evaluator = Evaluator(model, tokenizer, dataset)
metrics = evaluator.evaluate_model(trainer)
print(metrics)

6. Save Model
model.save_pretrained("./llama-bengali-lora")
tokenizer.save_pretrained("./llama-bengali-lora")
