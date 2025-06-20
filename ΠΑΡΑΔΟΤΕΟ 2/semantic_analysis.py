import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import numpy as np

# Βεβαιώσου ότι υπάρχει ο φάκελος ΠΑΡΑΔΟΤΕΟ 2
os.makedirs("ΠΑΡΑΔΟΤΕΟ 2", exist_ok=True)

# Φόρτωσε παραφρασμένα και αρχικά κείμενα
original_text = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes."

parrot_text = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."
vamsi_text = "Today is our dragon boat festival in our Chinese culture to celebrate it with all safe and great in our lives."
ramsri_text = "Today is our Chinese dragon boat festival, to celebrate it with all safe and great in our lives. Hope you too, enjoy it as my deepest wishes."

# Φτιάξε embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

texts = [original_text, parrot_text, vamsi_text, ramsri_text]
embeddings = model.encode(texts)

# Υπολογισμός cosine similarity
similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]

# Αποθήκευση αποτελεσμάτων σε txt
output_file = "ΠΑΡΑΔΟΤΕΟ 2/semantic_similarity_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Παραδοτέο 2: Υπολογισμός Ομοιότητας με Sentence Transformers\n")
    f.write("==========================================================\n\n")
    f.write(f"Original: {original_text}\n\n")
    f.write(f"Parrot: {parrot_text}\nSimilarity Score: {similarities[0]:.4f}\n\n")
    f.write(f"Vamsi: {vamsi_text}\nSimilarity Score: {similarities[1]:.4f}\n\n")
    f.write(f"Ramsri: {ramsri_text}\nSimilarity Score: {similarities[2]:.4f}\n")

# PCA για οπτικοποίηση
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings)

labels = ["Original", "Parrot", "Vamsi", "Ramsri"]
colors = ["blue", "green", "orange", "red"]

plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    plt.scatter(pca_embeddings[i, 0], pca_embeddings[i, 1], color=colors[i], label=label)
    plt.text(pca_embeddings[i, 0]+0.01, pca_embeddings[i, 1]+0.01, label)

plt.title("PCA Projection of Sentence Embeddings")
plt.legend()
plt.grid(True)

# Αποθήκευση του plot
plt.savefig("ΠΑΡΑΔΟΤΕΟ 2/semantic_pca_plot.png")
plt.close()
