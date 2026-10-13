import json
import logging
from pathlib import Path
import numpy as np
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from .config import (
    BERT_MODEL_NAME,
    SAVED_BERT_MODEL_DIR,
    CLASSIFICATION_TRAIN,
    CLASSIFICATION_VAL,
    LABELS,
    LABEL2ID,
    ID2LABEL,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    OUTPUTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(filepath: Path):
    """Loads records from a JSONL file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}. Run prepare_data.py first.")
    data = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_hf_dataset(raw_records):
    """Extracts text inputs and label IDs into a Hugging Face Dataset."""
    texts = []
    labels = []
    for item in raw_records:
        text = item.get("input", "").strip()
        label_str = item.get("output", "")
        if text and label_str in LABEL2ID:
            texts.append(text)
            labels.append(LABEL2ID[label_str])
    return Dataset.from_dict({"text": texts, "label": labels})


def compute_metrics(eval_pred):
    """Calculates Accuracy and Weighted F1-score for evaluation."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def train_parsbert():
    """Fine-tunes ParsBERT on the domain text classification dataset."""
    logger.info(f"Loading tokenizer & model: {BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    logger.info("Loading training and validation splits...")
    train_raw = load_jsonl(CLASSIFICATION_TRAIN)
    val_raw = load_jsonl(CLASSIFICATION_VAL)

    train_ds = prepare_hf_dataset(train_raw)
    val_ds = prepare_hf_dataset(val_raw)
    logger.info(f"Dataset ready. Train size: {len(train_ds)}, Validation size: {len(val_ds)}")

    def tokenize_func(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    train_tokenized = train_ds.map(tokenize_func, batched=True)
    val_tokenized = val_ds.map(tokenize_func, batched=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUTS_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting ParsBERT training...")
    trainer.train()

    eval_results = trainer.evaluate()
    logger.info(f"Final Evaluation Metrics: {eval_results}")

    SAVED_BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(SAVED_BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(SAVED_BERT_MODEL_DIR))
    logger.info(f"Fine-tuned ParsBERT model and tokenizer saved to {SAVED_BERT_MODEL_DIR}")


if __name__ == "__main__":
    train_parsbert()
