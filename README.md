# NLP Rewrite - Απαλλακτική Εργασία

Αυτό το repository περιέχει τα αρχεία της απαλλακτικής εργασίας για το μάθημα **Επεξεργασία Φυσικής Γλώσσας (NLP)**.

## 📂 Δομή του Repository

### Παραδοτέο 1: Παραφράσεις
Περιλαμβάνει:
- `paraphrase_parrot.py`, `paraphrased_parrot.txt` – Παραφράσεις με το μοντέλο Parrot.
- `paraphrase_vamsi.py`, `paraphrased_vamsi.txt` – Παραφράσεις με το μοντέλο Vamsi.
- `paraphrase_ramsri.py`, `paraphrased_ramsri.txt` – Παραφράσεις με το μοντέλο Ramsrigouthamg T5.
- `compare_models.py`, `semantic_comparison_results.txt` – Σύγκριση παραφράσεων (cosine similarity).

### Παραδοτέο 2: Υπολογιστική Ανάλυση
Περιλαμβάνει:
- `semantic_analysis.py` – Υπολογισμός cosine similarity και οπτικοποίηση με PCA.
- `semantic_similarity_analysis.txt` – Αποτελέσματα similarity.
- `semantic_pca_plot.png` – Οπτικοποίηση embeddings.

### Παραδοτέο 3: Δομημένη Αναφορά
Περιλαμβάνει:
- `paradoteo3.docx` – Ολοκληρωμένη αναφορά με περιγραφή μεθοδολογίας, αποτελέσματα πειραμάτων, πίνακες και διαγράμματα.

---

## 🔧 Οδηγίες Εκτέλεσης

1️⃣ **Κλωνοποιήστε το repository**
```bash
git clone <url_repo>
cd NLP_Rewrite
```

2️⃣ **Δημιουργήστε και ενεργοποιήστε περιβάλλον**
```bash
conda create -n nlp_project python=3.9
conda activate nlp_project
```
ή με venv:
```bash
python -m venv nlp_project
source nlp_project/bin/activate   # (Linux/Mac)
.
lp_project\Scriptsctivate    # (Windows)
```

3️⃣ **Εγκαταστήστε τις απαιτούμενες βιβλιοθήκες**
```bash
pip install -r requirements.txt
```

4️⃣ **Εκτελέστε τα αρχεία**
```bash
python Παραδοτέο1/paraphrase_parrot.py
python Παραδοτέο2/semantic_analysis.py
# κλπ
```

---

## 📝 Απαιτήσεις
- Python 3.9+
- PyTorch
- transformers
- sentence-transformers
- scikit-learn
- matplotlib

---

## ✉️ Επικοινωνία
Φοιτητής: **[Συμπληρώστε το όνομά σας]**