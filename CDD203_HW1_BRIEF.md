# CDD 203 HW1 — hERG Blocker Classifier: Build Brief

A brief for Claude Code. Read all of it before writing code.

---

## 0. Placeholders to fill before starting

Ask me for these if they are not already set:

- `[GROUP_NAMES]` — first and last names of all four group members, used in filenames and the README
- `[REPO_PATH]` — where the project lives locally
- `[HF_USERNAME]` — Hugging Face username for the Spaces URL

---

## 1. Goal

Build a binary classifier that predicts whether a molecule is a hERG blocker from its SMILES string, export it to ONNX, and deploy it as a public web app.

- `Y = 1` → hERG blocker
- `Y = 0` → non-blocker

hERG (human Ether-à-go-go-Related Gene) encodes a potassium channel. Blocking it causes QT prolongation and cardiotoxicity risk, so this is a drug-safety screening model.

---

## 2. Hard constraints (from the assignment — do not violate)

| Constraint | Limit |
|---|---|
| Environment | `environment.yml` → conda env `cdd203-hw1`, Python 3.10/3.11/3.12 only |
| Allowed extra libraries | ONLY `rdkit==2023.9.6`, `torch==2.2.2`, `transformers==4.46.3`. Nothing else without permission. **No XGBoost, no LightGBM, no TensorFlow.** |
| Device | CPU only. No `.cuda()`, no hard-coded `"mps"` |
| Model size | Under 150 million parameters (feature extraction and classification combined) |
| Training time | Under 30 minutes on a laptop CPU |
| ONNX file | At most 50 MB, committed directly to GitHub. No Git LFS |
| Pretrained models | Must be public on Hugging Face, loaded via `from_pretrained`, **without** `trust_remote_code` |
| Reproducibility | Set a random seed everywhere |

---

## 3. Part 1 — training script

File: `Part1_[GROUP_NAMES].py`, based on `Part1_TEMPLATE.py`. Four TODOs.

### TODO 1 — group member names

### TODO 2 — decision threshold (must not be 0.5)

**Selection method — do not hard-code a guess.** Compute it on the **validation set**, never the test set:

```python
from sklearn.metrics import roc_curve
import numpy as np

proba = model.predict_proba(X_valid)[:, 1]
fpr, tpr, thr = roc_curve(y_valid, proba)
spec = 1 - fpr

t_sens   = thr[np.argmax(tpr >= 0.90)]          # target sensitivity >= 0.90
t_youden = thr[np.argmax(tpr - fpr)]            # Youden's J
ok = spec > 0.7
t_comp   = thr[ok][np.argmax(tpr[ok])]          # max sensitivity, specificity > 0.7
```

Print sensitivity and specificity at each candidate, pick one, round to two decimals.

**Rationale to carry into the README:** a false negative (a real blocker called safe) lets a cardiotoxic compound advance through development at great cost, and potentially harms patients. A false positive only triggers a confirmatory lab hERG assay, which is cheap. The errors are asymmetric, so we accept more false positives to reduce false negatives — meaning a threshold **below** 0.5.

### TODO 3 — `featurize()`

**Requirement:** fixed-length output regardless of SMILES length. The held-out test set contains SMILES up to 200 characters, longer than anything in training.

**Design — fingerprint block + descriptor block, driven by the known hERG pharmacophore:**

| Known hERG blocker characteristic | Feature that encodes it |
|---|---|
| Basic tertiary nitrogen, protonated at physiological pH, cation-π with Tyr-652 | Basic-N SMARTS count, tertiary-amine count, Gasteiger max/min partial charge, formal charge |
| Lipophilic and bulky (membrane crossing, hydrophobic inner cavity fit) | `MolLogP`, `MolWt`, `TPSA`, `LabuteASA`, heavy atom count |
| Aromatic ring systems, π–π stacking | `NumAromaticRings`, aromatic atom fraction, ring count |
| Binds the large aqueous inner cavity of the pore domain | `NumRotatableBonds`, `FractionCSP3` |
| Structurally diverse scaffolds | Morgan/ECFP bits |

Build it as **four configurations behind a flag** so we can measure rather than guess:

1. **A** — Morgan ECFP4, 2048 bits, radius 2 (baseline)
2. **B** — A + the ~18 descriptors above
3. **C** — B + FCFP (feature-based Morgan, 1024) + atom-pair (1024) + pharmacophore distance histogram
4. **D** — C + ChemBERTa embeddings (optional experiment, see below)

**Why FCFP and atom-pair matter here:** FCFP abstracts atoms to roles (donor, acceptor, aromatic, cation), so a piperidine nitrogen and a piperazine nitrogen look alike — this generalizes across chemotypes, which matters because the held-out test set is "different in composition." Atom-pair encodes (atom type, atom type, topological distance) triples, which directly captures the distance between the cationic nitrogen and the aromatic rings — the actual pharmacophore geometry that plain Morgan cannot express.

**Pharmacophore distance histogram** (config C):

```python
dm = Chem.GetDistanceMatrix(mol)
n_idx   = [m[0] for m in mol.GetSubstructMatches(BASIC_N)]
aro_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
dists   = [dm[i][j] for i in n_idx for j in aro_idx]
# 10-bin histogram over bond distances 1-10, plus min/mean/max
```

The classic hERG pharmacophore places the cationic nitrogen roughly 5–7 bonds from an aromatic centroid.

**Basic nitrogen SMARTS** (sp3 N, excluding amides, nitro, already-charged):

```python
BASIC_N = Chem.MolFromSmarts(
    "[NX3;H0,H1,H2;!$(N[#6]=[O,N,S]);!$(N=*);!$(N#*);!$([N+])]")
```

**Robustness requirements:**
- Invalid SMILES → return a zero vector of the correct length, never crash
- `np.nan_to_num` all descriptors (Gasteiger can produce inf/nan)
- Disable RDKit logging
- Return `float32`

**Mandatory tests before submitting:**

```python
assert featurize(["C" * 200]).shape[1] == FEATURE_LEN     # 200-char SMILES
assert featurize(["not_a_molecule"]).shape[1] == FEATURE_LEN
assert featurize(["CCO"]).shape[1] == FEATURE_LEN
```

### TODO 4 — `build_model()`

Return an sklearn model so the template's ONNX export works.

**Candidates to compare — do not pick blind:**

```python
SEED = 42

rf = RandomForestClassifier(
    n_estimators=500, max_depth=20, min_samples_leaf=2,
    max_features="sqrt", class_weight="balanced_subsample",
    n_jobs=-1, random_state=SEED)

hgb = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=True, random_state=SEED)

et = ExtraTreesClassifier(
    n_estimators=500, max_depth=20, min_samples_leaf=2,
    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)

lr = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(C=1.0, max_iter=3000,
                       class_weight="balanced", random_state=SEED))
```

**Notes:**
- Class imbalance is likely. Use `class_weight="balanced"` / `balanced_subsample`; `sample_weight` for HistGradientBoosting. Set weighting **first**, then pick the threshold on the reweighted model.
- Watch the 50 MB ONNX cap. A deep 1000-tree forest on 2048 features can exceed it. Cap `n_estimators` at 300–500 and set `max_depth` or `min_samples_leaf`. **Check the exported file size and prune if needed.**
- Verify HistGradientBoosting actually converts through skl2onnx before committing to it.

---

## 4. Comparison harness (build this first)

Before finalizing anything, write a script that crosses **featurization configs A–D** with **models RF / HGB / ET / LR** and prints a table of **5-fold cross-validated AUROC** on the training set, plus mean inference time per molecule.

Use cross-validation, not a single split — the dataset is small and a single-split difference under ~0.02 AUROC is noise. Report mean ± std.

Then pick the winner by evidence, not by assumption.

**Realistic expectation to keep in mind:** the spread between a tuned RandomForest, HistGradientBoosting, and LogisticRegression on fingerprint data is usually small (0.02–0.04 AUROC). Featurization choice matters more than model choice. Config C is my prior for best accuracy-per-complexity, but the harness decides.

---

## 5. ChemBERTa experiment (config D — optional, measure before committing)

`DeepChem/ChemBERTa-77M-MLM` is 77M params (under the 150M cap). Mean-pool the last hidden state to a 384-dim embedding, concatenate with the fingerprint, train sklearn on top.

**Honest caveats — surface these in the comparison, do not hide them:**
- CPU inference is 50–100× slower than a fingerprint, conceding both speed competition categories
- The app must reproduce embeddings bit-identically or the 25% "app matches ONNX" check fails
- On a few-thousand-molecule dataset, ChemBERTa typically performs *comparably* to good fingerprints, not dramatically better. The literature is mixed. **This is not a reliable win.**

Only submit config D if cross-validated AUROC clearly beats config C.

---

## 6. Competition strategy

Six categories; only your **best ranking** counts (1st = +6%, 2nd = +4%, 3rd = +2%). So pick one target rather than compromising across all of them.

| Category | Strategy |
|---|---|
| Highest AUROC | Config C or D, HGB or RF |
| Highest sensitivity, specificity > 0.7 | Config C + threshold `t_comp` |
| Highest specificity, sensitivity > 0.7 | Mirror of the above |
| Highest precision, sensitivity > 0.7 | Tune threshold on precision |
| **Fastest with AUROC > 0.725** | **512-bit Morgan + ~15 descriptors + LogisticRegression** |
| **Fastest with AUROC > 0.85** | Small fingerprint, possibly top-200 bits by RF importance, LR or shallow HGB |

The two speed slots are arguably the easiest to place in, since most groups will default to 2048 bits in a random forest without considering inference time.

---

## 7. Part 2 — web app

Gradio on Hugging Face Spaces. Requirements:

1. Load the `.onnx` model on the backend (onnxruntime)
2. Text input for a SMILES string
3. **Real-time validation** — `Chem.MolFromSmiles` on every change; **Predict button disabled until valid**
4. Run **the exact same `featurize()`** as Part 1 — import it from a shared module, do not reimplement it
5. Output predicted class and probability, readable by a non-expert
6. Clean, uncluttered design (5% of the grade)

**The single biggest grading item is 25% for the app producing the same predictions as the submitted ONNX file.** Write an explicit test that runs a list of SMILES through both the local ONNX model and the app's prediction function and asserts the probabilities match.

**Wording:** label outputs "hERG blocker / non-blocker", **not** "safe / unsafe". A non-blocker only passes this one cardiac check — it says nothing about hepatotoxicity, other side effects, or efficacy. Don't let users over-trust a negative.

---

## 8. Deliverables (GitHub repo, link submitted on Canvas)

| # | Item | Notes |
|---|---|---|
| 1 | `Part1_[GROUP_NAMES].py` | Training script |
| 2 | `Part1_[GROUP_NAMES].onnx` | ≤ 50 MB, committed directly |
| 3 | `requirements.txt` | Any added libraries at the exact pinned versions |
| 4 | `Part 2.zip` | All app code **plus a copy of the same .onnx file** |
| 5 | `README.md` | App URL, all group member names, up to 3 paragraphs (threshold / features / model + hyperparameters), 1 paragraph on LLM use |

### README writeup content

- **(a) Threshold:** the selection rule used, the chosen value, and the asymmetric-cost reasoning from §3 TODO 2
- **(b) Features:** 2048-bit Morgan ECFP4 for scaffold diversity, plus descriptors mapped to the hERG pharmacophore — basic nitrogen counts and partial charge for the cation-π interaction with Tyr-652, logP and surface area for the lipophilicity and bulk driving membrane permeation and cavity binding, aromatic ring counts for π–π stacking. Plus FCFP and atom-pair if config C wins.
- **(c) Model and hyperparameters:** the winner, with cross-validated AUROC for each candidate so the choice is evidenced.

**Report the comparison honestly.** If the descriptor block or ChemBERTa does not improve validation AUROC, say so and submit the simpler model. Stating a measured null result is a stronger writeup than claiming a benefit that wasn't measured.

---

## 9. Final checks before submitting

```bash
conda env create -f environment.yml && conda activate cdd203-hw1
python Part1_[GROUP_NAMES].py
python TestModel.py Part1_[GROUP_NAMES].py      # every line must say PASS
```

- [ ] Fresh environment, both commands run clean
- [ ] Every `TestModel.py` line says PASS
- [ ] ONNX file ≤ 50 MB, no Git LFS
- [ ] `featurize()` handles 200-char SMILES and invalid SMILES
- [ ] Seeds set everywhere
- [ ] App predictions match ONNX predictions (tested, not assumed)
- [ ] Predict button disabled until SMILES is valid
- [ ] README has: app URL, all names, 3-paragraph writeup, LLM-use paragraph
- [ ] `Part 2.zip` contains app code and the matching `.onnx`
- [ ] No CUDA/MPS, no disallowed libraries

---

## 10. Working order

1. Confirm the placeholders in §0
2. Build `featurize()` with configs A–D + the robustness tests
3. Build the comparison harness (§4) and run it
4. Pick featurization + model from the results
5. Pick the threshold from validation (§3 TODO 2)
6. Finalize `Part1_[GROUP_NAMES].py`, export ONNX, check size
7. Run `TestModel.py`
8. Build the Gradio app importing the same `featurize()`
9. Write the parity test between app and ONNX
10. Deploy to Spaces, write the README, zip Part 2
