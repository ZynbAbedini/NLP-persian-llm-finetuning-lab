from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Root paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
RAW_CHUNKS_FILE = RAW_DATA_DIR / "sample_chunks.jsonl"
CLASSIFICATION_TRAIN = PROCESSED_DATA_DIR / "classification_train.jsonl"
CLASSIFICATION_VAL = PROCESSED_DATA_DIR / "classification_val.jsonl"
CLASSIFICATION_TEST = PROCESSED_DATA_DIR / "classification_test.jsonl"

QA_TRAIN = PROCESSED_DATA_DIR / "qa_train.jsonl"
QA_VAL = PROCESSED_DATA_DIR / "qa_val.jsonl"
QA_TEST = PROCESSED_DATA_DIR / "qa_test.jsonl"

# Model Configuration
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "HooshvareLab/bert-fa-base-uncased")
SAVED_BERT_MODEL_DIR = MODELS_DIR / "parsbert_classifier"

# Classification Labels
LABELS = ["هوش مصنوعی", "پردازش زبان طبیعی", "یادگیری ماشین", "عمومی"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Hyperparameters
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))
