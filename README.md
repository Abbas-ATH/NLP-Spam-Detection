# Robust English SMS Spam Detection Under Controlled Spelling Errors

**Course:** Natural Language Processing — Option 1, First Semester 2026–2027  
**Section:** 7AI02  
**Group:** 5  
**Instructor:** Mohammed Talal Ahmed Mouslly

**Team members**

- Abbas Abdulaziz Al Thunayan — 2240004112
- Mujtaba Haji Alnuwaysir — 2240007941
- Abdullah Haji Alnuwaysir — 2240005501

**Approved research question:** Which method detects misspelled spam messages more reliably?

## Project overview

This project compares two English SMS spam classifiers:

1. **Traditional method:** Word-level TF-IDF with Logistic Regression.
2. **Pretrained method:** Frozen `sentence-transformers/all-MiniLM-L6-v2` sentence embeddings with Logistic Regression.

Both methods are evaluated on the same held-out SMS messages, first in their original form and then with controlled spelling errors. The notebook accepts an unfamiliar message and returns **Spam** or **Not spam** without retraining the saved classifiers.

The repository contains the executed notebook, source code, public dataset files, saved classifiers, prediction tables, figures, a six-page report in PDF, and a separate team appendix. The PowerPoint presentation is being prepared separately for the final submission.

## Evaluation results

The held-out test set contains **781 messages**: **672 Not spam (ham)** and **109 Spam**. The classification threshold is fixed at **0.5**.

| Condition | Method | Accuracy | Macro F1 | Spam recall | False positives | False negatives |
|---|---|---:|---:|---:|---:|---:|
| Clean | TF-IDF | 97.44% | 0.9431 | 83.49% | 2 | 18 |
| Clean | MiniLM | 97.57% | 0.9467 | 85.32% | 3 | 16 |
| Deletion | TF-IDF | 97.31% | 0.9395 | 81.65% | 1 | 20 |
| Deletion | MiniLM | 97.70% | 0.9501 | 87.16% | 4 | 14 |
| Transposition | TF-IDF | 97.44% | 0.9427 | 82.57% | 1 | 19 |
| Transposition | MiniLM | 97.70% | 0.9497 | 86.24% | 3 | 15 |
| Mixed | TF-IDF | 97.31% | 0.9395 | 81.65% | 1 | 20 |
| Mixed | MiniLM | 97.70% | 0.9501 | 87.16% | 4 | 14 |

In the mixed condition, MiniLM identifies **95 of 109** spam messages, versus **89 of 109** for TF-IDF. This represents a **5.50 percentage-point** spam-recall difference, but MiniLM also generates **four false positives**, compared with **one** for TF-IDF.

The paired recall-difference bootstrap interval for mixed spam is **0.00–11.01 percentage points**, with an exploratory exact McNemar/binomial **p-value of 0.1094**. These results describe this test split; they do not establish universal superiority. See `results/conclusions.md` and the project report for the analysis and limitations.

## Setup and running the notebook

The recorded experiment used **Python 3.12.10 on Windows (CPU)**. Package requirements are listed in `requirements.txt`, and recorded runtime information is in `results/run_metadata.json`.

Start from the **repository root directory** (the folder containing `README.md`, `requirements.txt`, `src/`, and `notebooks/`). An initial setup requires access to Python package sources and Hugging Face to obtain the pretrained encoder. The public UCI dataset archive is already included under `data/raw/`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name nlp-spam --display-name "NLP Spam Project"
.\.venv\Scripts\python.exe -m jupyter lab
```

### Linux or macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name nlp-spam --display-name "NLP Spam Project"
.venv/bin/python -m jupyter lab
```

If installation of PyTorch requires platform-specific packages, consult the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

In JupyterLab:

1. Open `notebooks/Spam_Detection_Project.ipynb`.
2. Select the **NLP Spam Project** kernel.
3. Choose **Restart Kernel and Run All Cells**.
4. In **Try it**, enter a new SMS and select either classifier or both.

The notebook follows the instructor's required sections: **Setup, Data, Method 1, Method 2, Results, Try it, and Sources**. Its saved outputs were produced by a previous complete execution. A successful run in a fresh kernel was recorded for the original execution environment; a completely isolated installation on another machine has not been established.

### Optional command-line execution and verification

The following scripts are included in the repository:

```bash
python run_experiment.py
python verify_project.py
```

`run_experiment.py` executes the data preparation, training, testing, and results pipeline. **It retrains both classifiers and rewrites generated results**, so use it when you intend to reproduce the experiment.

`verify_project.py` checks saved predictions, metrics, splits, typo variants, and inference. **Run the notebook or `run_experiment.py` at least once before verification on a new machine**, because verification expects the pretrained MiniLM encoder weights to exist locally at `models/minilm_encoder/`. This encoder directory is not shipped in the repository.

## Dataset and leakage controls

The project uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), distributed under **CC BY 4.0**. Its original archive and README are retained in `data/raw/`. Each record is parsed at the first tab character, preserving the message text.

- **Original messages:** 5,574 (4,827 ham; 747 spam).
- **Exact duplicate messages removed:** 403.
- **Unique messages:** 5,171 (4,518 ham; 653 spam).
- **Train / development / test sizes:** 3,619 / 771 / 781.
- **Spam counts in each partition:** 473 / 71 / 109.
- **Random seed:** `20261116`.

Before splitting, case/whitespace and selected URL, email, and digit patterns are normalized **for grouping only**. Character 3–5-gram TF-IDF similarity (threshold **0.92**) is then used to link related messages. A stratified group-based 20-fold assignment allocates **14 folds to training, 3 to development, and 3 to testing**, keeping identified related messages together.

These grouping features do **not** enter either classifier. All learned classifier preprocessing uses training data only. Model settings are selected on clean development data; the test set is used for the final comparison. Lack of sender or campaign identifiers means that all semantic relationships cannot be ruled out.

## Models and saved inference

The selected TF-IDF pipeline uses unigrams and bigrams, sublinear term frequency, `min_df=1`, and **38,657** vocabulary terms. Both selected Logistic Regression classifiers use `C=8`, no class weighting, the `liblinear` solver, a maximum of **2,000** iterations, and a decision threshold of **0.5**.

The pretrained MiniLM encoder is frozen and produces normalized **384-dimensional** sentence embeddings. It uses a maximum sequence length of **256 wordpieces**, batch size **64**, and four CPU threads in the recorded experiment.

The trained classifiers are stored in:

- `models/tfidf_logreg.joblib`
- `models/minilm_logreg.joblib`

The encoder metadata is stored in `models/encoder_metadata.json`. It records the exact pretrained revision:

`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`

The **MiniLM encoder weights are not included** in this archive. On the first MiniLM run, the code downloads the pinned public model and stores it under `models/minilm_encoder/`. After a successful download, the local encoder can be reused offline. MiniLM weights are distributed upstream under **Apache 2.0**.

**Security:** Only load serialized `.joblib` files from trusted sources.

### Predict a new SMS without retraining

After installing dependencies, from the repository root:

```python
from src.project import SpamPredictor

predictor = SpamPredictor()
print(predictor.predict("Please bring your notebook to tomorrow's lab.", model="both"))
print(predictor.predict("You won a cash prize! Call now to claim it.", model="MiniLM"))
print(predictor.predict("   "))
```

Alternatively, use the **Try it** section in the notebook. The classifier displays **Spam** or **Not spam**; numerical scores are **uncalibrated classifier scores**, not guaranteed probabilities. Empty or whitespace-only inputs return a helpful message.

## Controlled spelling-error experiment

After selecting both model configurations on development data, the experiment builds three test conditions: **deletion**, **transposition**, and **mixed**.

- Each modified message receives **one** edit in one eligible word.
- Deletion removes one internal character; transposition swaps two unequal adjacent internal characters.
- Mixed randomly selects **one of those operations** per message; it does not apply both.
- Eligible words contain at least five ASCII letters and have `wordfreq` Zipf frequency at least **3.0**.
- Edits producing words with Zipf frequency at least **2.5** are rejected to reduce real-word substitutions.
- URLs, email addresses, and tokens containing digits are protected.
- The deterministic selection uses the seed, message row ID, and operation.

The saved test variants record the original and modified text, operation, edit positions, and whether a modification succeeded. Successful changes were made to **707 deletion**, **708 transposition**, and **707 mixed** test messages. All **109 test spam messages** were modified in every typo condition; **74** mixed-condition ham messages were unchanged and remain identified as such.

Both models are evaluated on **identical** test texts and labels, with full-set results and a matched clean-versus-modified eligible-cohort comparison. No perturbed test messages enter training or development. The lexical filtering procedure reduces some undesirable edits but cannot guarantee preserved meaning in every context.

## Repository contents

| Path | Contents |
|---|---|
| `notebooks/Spam_Detection_Project.ipynb` | Main executed notebook, figures, evaluation, and interactive prediction |
| `src/project.py` and `src/__init__.py` | Data preparation, splitting, model training, typo generation, evaluation, and inference |
| `run_experiment.py` | Optional script to reproduce the full experiment |
| `verify_project.py` | Optional saved-result and integrity checks; requires local encoder weights |
| `requirements.txt` | Python dependencies |
| `data/raw/` | Original UCI archive, extracted SMS file, and dataset README |
| `data/processed/` | Split manifest, typo test manifests, and conflict audit |
| `models/` | Saved classifiers and encoder metadata; pretrained encoder downloaded separately |
| `results/metrics.csv` | Evaluation metrics for the complete test set and eligible cohorts |
| `results/predictions/` | Message-level predictions and scores for both methods |
| `results/*development_grid.csv` | Development-set hyperparameter comparisons |
| `results/*selection.json` | Selected classifier configurations |
| `results/error_analysis.csv` | Interpreted misclassification examples |
| `results/conclusions.md` | Experiment interpretation and limitations |
| `results/figures/` | Data distribution, comparison plots, and confusion matrices |
| `results/` | Additional audits, verification records, and run metadata |
| `report/NLP Spam Detection Report Final.pdf` | Six-page academic report |
| `appendix/Team Appendix Group 5 .docx` | Separate team appendix |
| `ATTRIBUTION.md` | Data, model, software, licenses, and AI-assistance disclosures |
| `.gitignore` | Local environment, cache, and downloaded encoder exclusions |

## Submission status

**Group:** 5  
**Section:** 7AI02  
**Instructor:** Mohammed Talal Ahmed Mouslly

The technical project, report, and separate appendix are present in this package. The course also requires a **PowerPoint presentation** and **a GitHub repository link** in the final Blackboard submission. The final ZIP should be named **`Group5_Project.zip`** and contain the completed mandatory deliverables. A GitHub publication or Blackboard submission is **not claimed** by this README.

The team should verify that the appendix accurately describes each member's actual technical contribution and that the examples satisfy the instructor's rules. All members must be able to explain the implementation and results during the presentation and individual questions.

## Attribution and references

ChatGPT/Codex assisted with implementation, execution, debugging, analysis, and documentation. Results reported here are drawn from executed code and saved prediction files. Use of AI assistance and third-party resources is documented in `ATTRIBUTION.md`.

- Almeida, Hidalgo, and Yamakami, *Contributions to the Study of SMS Spam Filtering*, DocEng 2011: https://doi.org/10.1145/2034691.2034742
- UCI SMS Spam Collection: https://doi.org/10.24432/C5CC84
- Reimers and Gurevych, *Sentence-BERT*, EMNLP-IJCNLP 2019: https://aclanthology.org/D19-1410/
- Sentence Transformers MiniLM model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- `wordfreq` package and source notes: https://github.com/rspeer/wordfreq
