import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import SAVED_BERT_MODEL_DIR, BERT_MODEL_NAME, LABELS, ID2LABEL


def predict_topic(text: str, model_path=None):
    """
    Predicts the domain category of a given Persian text using the fine-tuned ParsBERT model.
    """
    target_dir = model_path or SAVED_BERT_MODEL_DIR

    if not (target_dir / "config.json").exists():
        print(f"⚠️ مدل آموزش‌دیده در {target_dir} یافت نشد. لطفاً ابتدا اسکریپت آموزش را اجرا کنید.")
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(target_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(target_dir))
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()

    label = ID2LABEL.get(pred_idx, "نامشخص")
    confidence = probs[pred_idx].item()

    all_scores = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(LABELS))}
    return {
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "scores": all_scores
    }


def main():
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        user_text = "الگوریتم‌های یادگیری تقویتی و شبکه‌های عصبی عمیق در بازی‌های رایانه‌ای نتایج خیره‌کننده‌ای دارند."

    print("=" * 60)
    print(f"متن ورودی:\n«{user_text}»")
    print("=" * 60)

    result = predict_topic(user_text)
    if result:
        print(f"🎯 دسته‌بندی پیش‌بینی‌شده: {result['predicted_label']} (اطمینان: {result['confidence']*100:.2f}%)")
        print("\nامتیاز تمام کلاس‌ها:")
        for cls_name, score in result["scores"].items():
            print(f" - {cls_name}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
