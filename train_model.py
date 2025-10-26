# -*- coding: utf-8 -*-
"""
Treinamento do Fretting-Transformer (T5 reduzido).
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from pathlib import Path

def load_data(processed_dir):
    """
    Carrega dados processados em formato Hugging Face.
    """
    data = []
    for file in processed_dir.glob("*.txt"):
        lines = file.read_text().split('\n')
        input_text = lines[0].replace("INPUT: ", "")
        output_text = lines[1].replace("OUTPUT: ", "")
        data.append({"input_text": input_text, "target_text": output_text})
    return Dataset.from_pandas(pd.DataFrame(data))

def train_model(data_dir):
    """
    Treina T5 reduzido.
    """
    dataset = load_data(data_dir)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Usar small como reduzido
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def tokenize_function(examples):
        inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
        targets = tokenizer(examples["target_text"], max_length=512, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained("./fretting_transformer")
    tokenizer.save_pretrained("./fretting_transformer")

if __name__ == "__main__":
    train_model(Path("processed_data"))