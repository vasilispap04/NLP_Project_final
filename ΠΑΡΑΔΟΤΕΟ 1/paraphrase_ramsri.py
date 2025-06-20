from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Κείμενα προς παραφράση
text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes."""
text2 = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn."""

# Φόρτωση μοντέλου και tokenizer
model_name = "ramsrigouthamg/t5_paraphraser"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Συνάρτηση παραφράσεων
def paraphrase(text):
    input_text = f"paraphrase: {text} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(
        input_ids,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Παραφράσεις
rephrased1 = paraphrase(text1)
rephrased2 = paraphrase(text2)

# Αποθήκευση σε αρχείο
with open("ΠΑΡΑΔΟΤΕΟ 1/paraphrased_ramsri.txt", "w", encoding="utf-8") as f:
    f.write("Παραδοτέο 1Β - Μοντέλο 3: Ramsrigouthamg T5 Paraphraser\n")
    f.write("=" * 40 + "\n\n")
    f.write("Original Text 1:\n" + text1 + "\n\n")
    f.write("Rephrased Text 1:\n" + rephrased1 + "\n\n")
    f.write("Original Text 2:\n" + text2 + "\n\n")
    f.write("Rephrased Text 2:\n" + rephrased2 + "\n")

print("✅ Το αρχείο paraphrased_ramsri.txt δημιουργήθηκε με επιτυχία.")
