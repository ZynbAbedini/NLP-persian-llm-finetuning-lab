import json
import random
import logging
from typing import List, Dict
from pathlib import Path

from .config import (
    RAW_CHUNKS_FILE,
    PROCESSED_DATA_DIR,
    CLASSIFICATION_TRAIN,
    CLASSIFICATION_VAL,
    CLASSIFICATION_TEST,
    QA_TRAIN,
    QA_VAL,
    QA_TEST,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_chunks(file_path: Path) -> List[Dict]:
    """Loads raw text chunks from a JSONL file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw chunks file not found at: {file_path}")

    chunks = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                chunks.append(json.loads(line_str))
    return chunks


def build_classification_dataset(chunks: List[Dict]) -> List[Dict]:
    """Generates labeled text classification examples based on topic titles."""
    data = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        title = meta.get("title") or chunk.get("title", "")
        text = chunk.get("text", "").strip()

        if not text or len(text) < 40:
            continue

        if "هوش مصنوعی" in title:
            label = "هوش مصنوعی"
        elif "پردازش زبان" in title:
            label = "پردازش زبان طبیعی"
        elif "یادگیری ماشین" in title:
            label = "یادگیری ماشین"
        else:
            label = "عمومی"

        data.append({
            "instruction": "این متن علمی را به یکی از شاخه‌های تخصصی علوم کامپیوتر طبقه‌بندی کن.",
            "input": text[:400],
            "output": label,
            "title": title
        })
    return data


def build_qa_dataset(chunks: List[Dict]) -> List[Dict]:
    """Generates question-answering instruction pairs for LLM fine-tuning."""
    data = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        title = meta.get("title") or chunk.get("title", "این موضوع")
        text = chunk.get("text", "").strip()

        if not text or len(text) < 80:
            continue

        question = f"درباره «{title}» چه توضیحی در متن داده شده است؟"
        summary_answer = text[:250] + ("..." if len(text) > 250 else "")

        data.append({
            "messages": [
                {"role": "system", "content": "تو یک دستیار پژوهشی متخصص زبان فارسی هستی. به سؤالات بر اساس متن پاسخ دقیق بده."},
                {"role": "user", "content": f"متن مرجع:\n{text[:450]}\n\nپرسش: {question}"},
                {"role": "assistant", "content": summary_answer}
            ],
            "instruction": "بر اساس متن پاسخ بده.",
            "input": f"متن: {text[:450]}\nسؤال: {question}",
            "output": summary_answer
        })
    return data


def split_dataset(data: List[Dict], train_ratio: float = 0.70, val_ratio: float = 0.15) -> Dict[str, List[Dict]]:
    """Splits data randomly into Train, Validation, and Test sets."""
    random.seed(42)
    shuffled = list(data)
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:]
    }


def save_jsonl(data: List[Dict], filepath: Path) -> None:
    """Saves a list of dictionaries to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_data_preparation() -> None:
    """Prepares and saves both classification and QA datasets."""
    logger.info(f"Loading raw chunks from {RAW_CHUNKS_FILE}...")
    chunks = load_raw_chunks(RAW_CHUNKS_FILE)
    logger.info(f"Loaded {len(chunks)} raw chunks.")

    # 1. Classification Data
    cls_data = build_classification_dataset(chunks)
    cls_splits = split_dataset(cls_data)
    save_jsonl(cls_splits["train"], CLASSIFICATION_TRAIN)
    save_jsonl(cls_splits["validation"], CLASSIFICATION_VAL)
    save_jsonl(cls_splits["test"], CLASSIFICATION_TEST)
    logger.info(f"Classification splits saved: Train={len(cls_splits['train'])}, Val={len(cls_splits['validation'])}, Test={len(cls_splits['test'])}")

    # 2. QA Data (for Unsloth / LoRA)
    qa_data = build_qa_dataset(chunks)
    qa_splits = split_dataset(qa_data)
    save_jsonl(qa_splits["train"], QA_TRAIN)
    save_jsonl(qa_splits["validation"], QA_VAL)
    save_jsonl(qa_splits["test"], QA_TEST)
    logger.info(f"QA splits saved: Train={len(qa_splits['train'])}, Val={len(qa_splits['validation'])}, Test={len(qa_splits['test'])}")


if __name__ == "__main__":
    run_data_preparation()
