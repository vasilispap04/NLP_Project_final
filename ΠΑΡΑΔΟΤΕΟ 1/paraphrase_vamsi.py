from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Επιλογή μοντέλου παραφράσεων
model_name = "Vamsi/T5_Paraphrase_Paws"

# Φορτώνουμε tokenizer και μοντέλο (σημαντικό: use_fast=False!)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Είσοδος παραδείγματος (βάλε εδώ το δικό σου κείμενο)
original_sentences = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn."
]

# Παραφραστική συνάρτηση
def paraphrase(sentence):
    input_text = f"paraphrase: {sentence} </s>"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt", padding="longest", truncation=True)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Εκτέλεση παραφράσεων
rephrased_sentences = [paraphrase(sent) for sent in original_sentences]

# Αποθήκευση σε αρχείο
with open("ΠΑΡΑΔΟΤΕΟ 1/paraphrased_vamsi.txt", "w", encoding="utf-8") as f:
    f.write("Παραδοτέο 1B - Μοντέλο 2: Vamsi\n")
    f.write("=" * 40 + "\n\n")
    for i, (orig, rephrased) in enumerate(zip(original_sentences, rephrased_sentences), 1):
        f.write(f"Original Text {i}:\n{orig}\n\n")
        f.write(f"Rephrased Text {i}:\n{rephrased}\n\n")

print("✅ Το αρχείο paraphrased_vamsi.txt δημιουργήθηκε με επιτυχία.")
