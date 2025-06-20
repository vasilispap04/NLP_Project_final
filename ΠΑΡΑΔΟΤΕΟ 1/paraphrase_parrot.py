from parrot import Parrot
import torch

# Κείμενα εισόδου
text1 = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes."
text2 = "During our final discuss, I told him about the new submission — the one we were waiting since last autumn."

# Αρχικοποίηση Parrot
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

def get_paraphrases(text):
    paraphrases = parrot.augment(input_phrase=text, max_return_phrases=3, max_length=60)
    if paraphrases:
        return [p[0] for p in paraphrases]
    return ["(No paraphrase generated)"]

# Παράφραση
rephrased1 = get_paraphrases(text1)
rephrased2 = get_paraphrases(text2)

# Αποθήκευση αποτελεσμάτων
with open("ΠΑΡΑΔΟΤΕΟ 1/paraphrased_parrot.txt", "w", encoding="utf-8") as f:
    f.write("Παραδοτέο 1Β - Μοντέλο 4: Parrot Paraphraser\n")
    f.write("=" * 40 + "\n\n")

    f.write("Original Text 1:\n" + text1 + "\n\n")
    f.write("Rephrased Text 1:\n" + "\n".join(rephrased1) + "\n\n")

    f.write("Original Text 2:\n" + text2 + "\n\n")
    f.write("Rephrased Text 2:\n" + "\n".join(rephrased2) + "\n")

print("✅ Το αρχείο paraphrased_parrot.txt δημιουργήθηκε με επιτυχία.")
