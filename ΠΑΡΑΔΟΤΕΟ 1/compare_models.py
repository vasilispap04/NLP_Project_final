from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Read paraphrased outputs from all three models
def read_text_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() and not line.lower().startswith("original"):
                return line.strip()

# Load rephrased text from each model
rephrased_parrot = read_text_from_file("ΠΑΡΑΔΟΤΕΟ 1/paraphrased_parrot.txt")
rephrased_vamsi = read_text_from_file("ΠΑΡΑΔΟΤΕΟ 1/paraphrased_vamsi.txt")
rephrased_ramsri = read_text_from_file("ΠΑΡΑΔΟΤΕΟ 1/paraphrased_ramsri.txt")

# Compute pairwise similarities
similarity_parrot_vamsi = util.cos_sim(model.encode(rephrased_parrot), model.encode(rephrased_vamsi)).item()
similarity_parrot_ramsri = util.cos_sim(model.encode(rephrased_parrot), model.encode(rephrased_ramsri)).item()
similarity_vamsi_ramsri = util.cos_sim(model.encode(rephrased_vamsi), model.encode(rephrased_ramsri)).item()

# Save results
with open("ΠΑΡΑΔΟΤΕΟ 1/semantic_comparison_results.txt", "w", encoding="utf-8") as f:
    f.write("Semantic Similarity Comparison between Parrot, Vamsi, and Ramsri\n")
    f.write("==============================================================\n")
    f.write(f"Parrot vs Vamsi:  {similarity_parrot_vamsi:.4f}\n")
    f.write(f"Parrot vs Ramsri: {similarity_parrot_ramsri:.4f}\n")
    f.write(f"Vamsi vs Ramsri:  {similarity_vamsi_ramsri:.4f}\n")

print("\n✅ Ολοκληρώθηκε η σύγκριση και αποθηκεύτηκε στο 'semantic_comparison_results.txt'")