import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load and preprocess your dataset
dataset = TextDataset(tokenizer=tokenizer, file_path="./data/dataset.txt", block_size=128)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-gpt2",
    overwrite_output_dir=True,
    num_train_epochs=180,  # Increased number of epochs
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,  # Adjusted learning rate
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model_save_path = "./fine-tuned-gpt2"
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)









