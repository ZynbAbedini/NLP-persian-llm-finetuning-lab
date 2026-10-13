# Persian LLM & Transformer Fine-Tuning Lab 🔬⚡

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/Unsloth-5x%20Faster-blue?style=for-the-badge" alt="Unsloth" />
  <img src="https://img.shields.io/badge/PEFT-QLoRA%204bit-orange?style=for-the-badge" alt="QLoRA" />
  <img src="https://img.shields.io/badge/Model-ParsBERT%20%7C%20Qwen%202.5-purple?style=for-the-badge" alt="Models" />
  <img src="https://img.shields.io/badge/Export-GGUF%20(Q4_K_M)-darkgreen?style=for-the-badge" alt="GGUF" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

A comprehensive, production-ready lab for fine-tuning Persian language models. Covers both **Encoder fine-tuning** (ParsBERT for domain text classification using Hugging Face `Trainer` API) and **Generative LLM fine-tuning** (Qwen 2.5 7B Instruct using **Unsloth & QLoRA** in Google Colab, with automated export to **GGUF** for local inference with Ollama and llama.cpp).

---

## 🇬🇧 English Documentation

### Key Highlights
* **Automated Dataset Engine:** Converts raw unstructured text into structured Train/Validation/Test splits for both multi-class text classification and instruction-following/QA formats.
* **ParsBERT Domain Fine-Tuning:** Fine-tunes `HooshvareLab/bert-fa-base-uncased` with Hugging Face's `Trainer` API, dynamic padding, and weighted F1 / Accuracy metrics.
* **Qwen 2.5 (7B) QLoRA Fine-Tuning:** Implements 4-bit quantized LoRA adaptation via **Unsloth**, reducing VRAM usage by 70% and accelerating training speed by 5x on Google Colab T4 GPUs.
* **Quantized GGUF Export:** Converts the fine-tuned 7B model directly into `q4_k_m` GGUF binaries, enabling offline CPU/GPU inference via Ollama and llama.cpp.
* **Inference CLI:** Ready-to-use prediction script with softmax confidence scores and class probability breakdown.

### Pipeline Architecture
```
[Raw Persian Chunks]
        │
        ├──► [prepare_data.py] ──► [Classification: Train/Val/Test] ──► [train_bert.py (ParsBERT)] ──► [predict.py]
        │
        └──► [QA / ChatML: Train/Val] ──► [qwen2_5_unsloth_qlora.ipynb] ──► [GGUF Export] ──► [Local Ollama]
```

### Project Structure
```text
persian-llm-finetuning-lab/
├── data/
│   ├── raw/                   # Raw scraped chunks
│   └── processed/             # Train/Val/Test splits (Classification & QA)
├── notebooks/
│   └── qwen2_5_unsloth_qlora.ipynb  # Unsloth + QLoRA Colab notebook with GGUF export
├── src/
│   ├── config.py              # Hyperparameters, paths, and label mappings
│   ├── prepare_data.py        # Dataset creation and splitting pipeline
│   ├── train_bert.py          # ParsBERT sequence classification training
│   └── predict.py             # CLI inference and prediction script
├── models/                    # Saved fine-tuned checkpoints
├── requirements.txt           # Production dependencies
└── .env.example               # Training parameters template
```

### Quickstart

1. **Clone & Setup Virtual Environment:**
   ```bash
   git clone https://github.com/your-username/persian-llm-finetuning-lab.git
   cd persian-llm-finetuning-lab
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Generate Supervised Datasets:**
   ```bash
   python -m src.prepare_data
   ```

3. **Train ParsBERT Classifier:**
   ```bash
   python -m src.train_bert
   ```

4. **Test Predictions on Custom Text:**
   ```bash
   python -m src.predict "الگوریتم‌های یادگیری ماشین در بینایی ماشین کاربرد دارند."
   ```

5. **Fine-Tune 7B LLM (Qwen 2.5):**
   * Open `notebooks/qwen2_5_unsloth_qlora.ipynb` in [Google Colab](https://colab.research.google.com).
   * Upload `data/processed/qa_train.jsonl` and `qa_val.jsonl`.
   * Run the notebook on a free T4 GPU and download your fine-tuned `.gguf` model!

---

## 🇮🇷 راهنمای فارسی (Persian Documentation)

### نکات کلیدی و ویژگی‌ها
* **خط‌لوله ساخت خودکار داده‌ها:** پردازش متون خام و تبدیل به دو فرمت مجزا: دسته‌بندی موضوعی چندکلاسه و فرمت محاوره‌ای (ChatML) برای مدل‌های زبانی.
* **فاین‌تیونینگ اختصاصی پارس‌برت (ParsBERT):** آموزش مدل `HooshvareLab/bert-fa-base-uncased` با کلاس `Trainer` هاگینگ‌فیس و ارزیابی کمّی با متریک‌های استاندارد آماری Accuracy و Weighted F1.
* **فاین‌تیونینگ مدل ۷ میلیاردی Qwen 2.5:** بازآموزی با متد پیشرفته **QLoRA** در گوگل کولب با استفاده از کتابخانه شتاب‌دهنده **Unsloth** (کاهش ۷۰ درصدی مصرف VRAM و ۵ برابر سرعت بیشتر).
* **خروجی فرمت بهینه GGUF:** تبدیل مستقیم مدل به نسخه فشرده ۴ بیتی کوانتایز شده جهت اجرای آفلاین روی لپ‌تاپ معمولی با Ollama و Llama.cpp.
* **اسکریپت پیش‌بینی و استنتاج (CLI):** ابزار تست سریع برای ارزیابی جملات جدید به همراه نمایش درصد اطمینان (Confidence).

### مراحل اجرا

۱. **نصب پکیج‌ها:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

۲. **تولید دیتاست‌های آموزش و اعتبارسنجی:**
```bash
python -m src.prepare_data
```

۳. **آموزش مدل پارس‌برت:**
```bash
python -m src.train_bert
```

۴. **تست و ارزیابی متن دلخواه:**
```bash
python -m src.predict "شبکه‌های عصبی پیچشی در پردازش تصویر دقت بالایی دارند."
```

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
