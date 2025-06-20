from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Φόρτωση μοντέλου και tokenizer
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Προτάσεις προς παραφράση
sentences = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."
]

# Δημιουργία αρχείου εξόδου στο φάκελο NLPProject
with open("paraphrased_output.txt", "w", encoding="utf-8") as f:
    f.write("Παραδοτέο 1Α - Ανακατασκευή Προτάσεων\n")
    f.write("===================================\n\n")
    for sentence in sentences:
        input_text = f"paraphrase: {sentence} </s>"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(
            input_ids,
            max_length=100,
            num_beams=5,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Εγγραφή στο αρχείο
        f.write("Original: " + sentence + "\n")
        f.write("Rephrased: " + paraphrased_text + "\n\n")

print("✅ Το αρχείο 'paraphrased_output.txt' δημιουργήθηκε μέσα στο φάκελο NLPProject.")
