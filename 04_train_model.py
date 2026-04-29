from google.colab import drive
drive.mount('/content/drive')

from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import BertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

config = BertConfig(
    vocab_size=60000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512
)
model = BertForMaskedLM(config)

# Load the trained tokenizer
tokenizer = BertTokenizerFast.from_pretrained("/content/drive/MyDrive/Tokenizer")

dataset = load_dataset(
    "text",
    data_files={
        "train": "/content/drive/MyDrive/Dataset/train.txt",
        "test": "/content/drive/MyDrive/Dataset/test.txt"
    })
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length"
    )
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.25
)

training_args = TrainingArguments(
    output_dir="./algerian_training",
    eval_strategy = "steps",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    learning_rate=5e-5,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=5000,
    save_steps=5000,
    save_total_limit=2,
)

#  trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

trainer.save_model("/content/drive/MyDrive/DziriML")

# train with another paramertre

config = BertConfig(
    vocab_size=60000,
    hidden_size=384,        # ↓ reduce
    num_hidden_layers=6,    # ↓ reduce
    num_attention_heads=6,
    max_position_embeddings=256
)
model = BertForMaskedLM(config)

# Load the trained tokenizer
tokenizer = BertTokenizerFast.from_pretrained("/content/drive/MyDrive/Tokenizer")

dataset = load_dataset(
    "text",
    data_files={
        "train": "/content/drive/MyDrive/Dataset/train.txt",
        "test": "/content/drive/MyDrive/Dataset/test.txt"
    })
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./DziriLM_02",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    fp16=True,
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

#  trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("DziriLM_02")