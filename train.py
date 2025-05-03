from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq

model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the data
dataset = load_dataset("json", data_files="train_data_richer.json", split="train")

# Tokenize
def tokenize(example):
    input = "Symptom: " + example["instruction"]
    target = example["response"]
    inputs = tokenizer(input, padding="max_length", truncation=True, max_length=256)
    targets = tokenizer(target, padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

# Training setup
args = TrainingArguments(
    output_dir="./petcare-flan-model",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()
trainer.save_model("petcare-model")
tokenizer.save_pretrained("petcare-model")
