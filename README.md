## Fine-Tuning LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations

## Project Overview

This project fine-tunes **LLaMA 3.1-8B-Instruct** to generate **empathetic responses in Bengali**.

Because the model is very large, we use **LoRA (Low-Rank Adaptation)** to fine-tune it efficiently on **free GPU environments like Kaggle**, without reducing the sequence length.

---

## Goals

- Train an LLM to respond **empathetically in Bengali**
- Use **parameter-efficient fine-tuning (LoRA)**
- Keep **full sequence length** (no truncation)
- Run successfully on **Kaggle free GPU**

---

## Key Features

- Clean and preprocess Bengali conversation data  
- Instruction-style formatting for LLM fine-tuning  
- LoRA-based fine-tuning (**attention layers only**)  
- Evaluation using:
  - **Perplexity**
  - **BLEU**
  - **ROUGE**
  - **Human-readable empathy samples**
- Proper experiment and response logging  

---

## Project Structure

### DatasetProcessor

Handles:
- Data loading  
- Cleaning  
- Formatting  
- Train / validation / test splitting  

### LLAMAFineTuner

- Loads the model  
- Applies LoRA  
- Trains the model  
- Saves LoRA adapters  

### Evaluator

- Computes evaluation metrics  
- Logs generated responses  

---

## Requirements

- **Kaggle GPU environment**
- **Python 3.10+**
- Hugging Face account with access to  
  **meta-llama/Meta-Llama-3.1-8B-Instruct**

### Libraries

- `transformers`
- `accelerate`
- `peft`
- `datasets`
- `evaluate`
- `sentence-transformers`
- `bitsandbytes`

---

## Installation

```bash
pip install -q --upgrade bitsandbytes==0.41.0
pip install -q --upgrade transformers accelerate peft datasets evaluate sentence-transformers
```

**Important:**  
Restart the Kaggle kernel after installing `bitsandbytes`.

---

## Hugging Face Token Setup

Set your token directly in the notebook:

```python
HF_TOKEN = "your_huggingface_token_here"
```

### Required Token Permissions

- ✅ Read access to public repositories  
- ✅ Read access to public gated repositories  

---

## Dataset Preprocessing

### What Happens Here

- Load CSV file (`Questions`, `Answers`)
- Normalize Bengali text
- Remove very short or duplicate entries
- Convert into **instruction-style format**
- Split dataset into:
  - **80% Train**
  - **10% Validation**
  - **10% Test**

### Example Instruction Format

```text
System: You are an empathetic counselor. Respond in Bengali.
User: <Question>
Assistant: <Answer>
```

---

## Model Loading (Kaggle Safe)

### Why 8-bit?

- **4-bit often breaks on Kaggle** due to `bitsandbytes` issues  
- **8-bit + LoRA** works reliably on free GPUs  

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_fast=True
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
```

---

## Applying LoRA

### Why LoRA?

- Trains only a **small number of parameters**
- Saves GPU memory
- Faster training

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

---

## Training

### Training Strategy

- Gradient checkpointing  
- Mixed precision (**FP16**)  
- Full sequence length (**no truncation**)  
- Gradient accumulation for effective batch size  

```python
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
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=8192,
    args=training_args
)

trainer.train()
```

---

## Evaluation

### Automatic Metrics

- **Perplexity** (from validation loss)
- **BLEU**
- **ROUGE-L**

### Human Evaluation

- Prints sample responses
- Manual empathy rating (**1–5**)

```python
evaluator = Evaluator(model, tokenizer, dataset)
metrics = evaluator.evaluate_model(trainer)
print(metrics)
```

---

## Logging

### LLAMAExperiments.jsonl

Stores:
- Experiment ID
- Model name
- LoRA configuration
- Training loss
- Validation loss
- Evaluation metrics
- Timestamp

### GeneratedResponses.jsonl

Stores:
- Input text
- Generated response
- Experiment ID
- Timestamp

---

## Saving the Model

```python
model.save_pretrained("./llama-bengali-lora")
tokenizer.save_pretrained("./llama-bengali-lora")
```

Only **LoRA adapters** are saved — the base model remains unchanged.

---

## Important Notes

- Kaggle free GPU cannot reliably run **4-bit**
- **8-bit + LoRA** is stable and recommended
- Full sequence length is preserved
- This setup is suitable for **research papers, internships, and academic evaluation**
