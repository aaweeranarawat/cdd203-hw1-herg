"""
Comparison harness: featurization configs A-D x models RF / HGB / ET / LR.

Prints 5-fold cross-validated AUROC (mean +/- std) on the training data plus the
mean inference time per molecule (featurize + predict), then names the winner.

    python test.py                 # configs A-C (D is skipped if torch/transformers are missing)
    python test.py --configs ABCD  # include the ChemBERTa experiment
    python test.py --quick         # 600-molecule subsample, 3 folds, to smoke-test the script

Configs
    A  Morgan ECFP4, 2048 bits, radius 2
    B  A + 18 hERG-pharmacophore descriptors
    C  B + FCFP (1024) + atom-pair (1024) + basic-N -> aromatic distance histogram
    D  C + ChemBERTa-77M-MLM mean-pooled embedding (384)
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
np.random.seed(SEED)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- featurization
N_ECFP, N_FCFP, N_AP = 2048, 1024, 1024
N_DESC = 18
N_PHARM = 13  # 10-bin histogram over bond distances 1-10, plus min / mean / max
N_CHEMBERTA = 384

BASIC_N = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N[#6]=[O,N,S]);!$(N=*);!$(N#*);!$([N+])]")
TERTIARY_N = Chem.MolFromSmarts("[NX3;H0;!$(N[#6]=[O,N,S]);!$(N=*);!$(N#*);!$([N+])]")

_chooser = rdMolStandardize.LargestFragmentChooser()
_ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_ECFP)
_fcfp = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=N_FCFP, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
_ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_AP)


def _prep(smi):
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else _chooser.choose(mol)


def _descriptors(mol):
    """18 descriptors mapped to the hERG pharmacophore."""
    try:
        mol.UpdatePropertyCache(strict=False)
        from rdkit.Chem import AllChem
        AllChem.ComputeGasteigerCharges(mol)
        q = np.array([float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()])
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        q = np.zeros(1)
    n_heavy = max(mol.GetNumHeavyAtoms(), 1)
    return [
        len(mol.GetSubstructMatches(BASIC_N)),                       # basic-N count (cation-pi, Tyr-652)
        len(mol.GetSubstructMatches(TERTIARY_N)),                    # tertiary amines
        q.max(), q.min(),                                            # Gasteiger max / min partial charge
        sum(a.GetFormalCharge() for a in mol.GetAtoms()),            # net formal charge
        Descriptors.MolLogP(mol), Descriptors.MolWt(mol),            # lipophilicity, bulk
        rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcLabuteASA(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumAromaticRings(mol),                  # pi-pi stacking
        sum(a.GetIsAromatic() for a in mol.GetAtoms()) / n_heavy,    # aromatic atom fraction
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),                 # flexibility for the pore cavity
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
    ]


def _pharm_hist(mol):
    """Distances (in bonds) between each basic N and each aromatic atom."""
    n_idx = [m[0] for m in mol.GetSubstructMatches(BASIC_N)]
    aro_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not n_idx or not aro_idx:
        return np.zeros(N_PHARM)
    dm = Chem.GetDistanceMatrix(mol)
    d = np.array([dm[i][j] for i in n_idx for j in aro_idx])
    hist, _ = np.histogram(d, bins=np.arange(0.5, 11.5, 1.0))
    return np.concatenate([hist / len(d), [d.min(), d.mean(), d.max()]])


def compute_blocks(smiles):
    """Every feature block for every molecule; configs are assembled by concatenation.
    Invalid SMILES give zero blocks of the correct length, never an exception."""
    blocks = {k: [] for k in ("ecfp", "desc", "fcfp", "ap", "pharm")}
    sizes = {"ecfp": N_ECFP, "desc": N_DESC, "fcfp": N_FCFP, "ap": N_AP, "pharm": N_PHARM}
    for s in smiles:
        mol = _prep(s)
        if mol is None:
            row = {k: np.zeros(n) for k, n in sizes.items()}
        else:
            row = {"ecfp": _ecfp.GetFingerprintAsNumPy(mol), "desc": np.asarray(_descriptors(mol)),
                   "fcfp": _fcfp.GetFingerprintAsNumPy(mol), "ap": _ap.GetFingerprintAsNumPy(mol),
                   "pharm": _pharm_hist(mol)}
        for k in blocks:
            blocks[k].append(np.nan_to_num(np.asarray(row[k], np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    return {k: np.vstack(v).astype(np.float32) for k, v in blocks.items()}


_BERT = {}


def chemberta_embed(smiles, batch=32):
    """Mean-pooled last hidden state of DeepChem/ChemBERTa-77M-MLM (384-dim). CPU only."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not _BERT:
        torch.manual_seed(SEED)
        name = "DeepChem/ChemBERTa-77M-MLM"
        _BERT["tok"], _BERT["model"] = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            enc = _BERT["tok"](list(smiles[i:i + batch]), padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            h = _BERT["model"](**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    return np.nan_to_num(np.vstack(out)).astype(np.float32)


CONFIGS = {"A": ["ecfp"], "B": ["ecfp", "desc"], "C": ["ecfp", "desc", "fcfp", "ap", "pharm"],
           "D": ["ecfp", "desc", "fcfp", "ap", "pharm", "bert"]}


def assemble(config, blocks):
    return np.hstack([blocks[b] for b in CONFIGS[config]]).astype(np.float32)


def featurize(smiles, config="C"):
    """Stand-alone featurizer for one config (what Part1_*.py will eventually ship)."""
    blocks = compute_blocks(smiles)
    if "bert" in CONFIGS[config]:
        blocks["bert"] = chemberta_embed(smiles)
    return assemble(config, blocks)


def self_test():
    for cfg in "ABC":
        n = featurize(["CCO"], cfg).shape[1]
        assert featurize(["C" * 200], cfg).shape[1] == n
        assert featurize(["not_a_molecule"], cfg).shape[1] == n
        assert not featurize(["not_a_molecule"], cfg).any()
        assert featurize(["CCO"], cfg).dtype == np.float32


# ---------------------------------------------------------------------- models
def make_models():
    return {
        "RF": RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2,
                                     max_features="sqrt", class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=SEED),
        "HGB": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                              l2_regularization=1.0, early_stopping=True,
                                              random_state=SEED),
        "ET": ExtraTreesClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "LR": make_pipeline(StandardScaler(with_mean=False),
                            LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                                               random_state=SEED)),
    }


def fit(name, model, X, y):
    if name == "HGB":  # no class_weight argument -> balanced sample_weight
        return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)


# --------------------------------------------------------------------- harness
def load_data(use_all):
    df = pd.read_csv(HERE / "data" / "hw1data.tab", sep="\t")
    df.columns = ["Drug_ID", "Drug", "Y"]
    if not use_all:  # keep the template's 20% test split untouched
        df = df.drop(df.sample(frac=0.2, random_state=42).index)
    return df["Drug"].tolist(), df["Y"].to_numpy()


def time_featurization(smiles, config, n=200):
    """Mean seconds per molecule to featurize (blocks needed by this config only)."""
    sample = smiles[:n]
    if config != "D":
        return _timed_blocks(sample, CONFIGS[config]) / len(sample)
    sec = _timed_blocks(sample, CONFIGS["C"])
    t = time.perf_counter()
    chemberta_embed(sample[:50])
    return (sec + (time.perf_counter() - t) / 50 * len(sample)) / len(sample)


def _timed_blocks(sample, needed):
    t = time.perf_counter()
    for s in sample:
        mol = _prep(s)
        if mol is None:
            continue
        if "ecfp" in needed:
            _ecfp.GetFingerprintAsNumPy(mol)
        if "desc" in needed:
            _descriptors(mol)
        if "fcfp" in needed:
            _fcfp.GetFingerprintAsNumPy(mol)
        if "ap" in needed:
            _ap.GetFingerprintAsNumPy(mol)
        if "pharm" in needed:
            _pharm_hist(mol)
    return time.perf_counter() - t


def run(args):
    smiles, y = load_data(args.all)
    if args.quick:
        rng = np.random.RandomState(SEED)
        keep = rng.choice(len(smiles), min(600, len(smiles)), replace=False)
        smiles, y = [smiles[i] for i in keep], y[keep]
        args.folds = 3
    print("molecules: %d   blocker rate: %.3f   folds: %d" % (len(y), y.mean(), args.folds))

    t = time.perf_counter()
    blocks = compute_blocks(smiles)
    print("fingerprint/descriptor blocks computed in %.1fs" % (time.perf_counter() - t))

    configs = []
    for c in args.configs:
        if c == "D":
            try:
                import torch, transformers  # noqa: F401
            except ImportError:
                print("config D skipped: torch / transformers not installed in this env")
                continue
            t = time.perf_counter()
            blocks["bert"] = chemberta_embed(smiles)
            print("ChemBERTa embeddings computed in %.1fs" % (time.perf_counter() - t))
        configs.append(c)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    rows = []
    for cfg in configs:
        X = assemble(cfg, blocks)
        feat_s = time_featurization(smiles, cfg)
        for mname in args.models:
            aucs, pred_s = [], []
            for tr, te in skf.split(X, y):
                model = fit(mname, make_models()[mname], X[tr], y[tr])
                t = time.perf_counter()
                p = model.predict_proba(X[te])[:, 1]
                pred_s.append((time.perf_counter() - t) / len(te))
                aucs.append(roc_auc_score(y[te], p))
            rows.append({"config": cfg, "model": mname, "n_features": X.shape[1],
                         "auroc_mean": np.mean(aucs), "auroc_std": np.std(aucs),
                         "feat_ms": feat_s * 1e3, "pred_ms": np.mean(pred_s) * 1e3})
            r = rows[-1]
            print("  %s/%-3s AUROC %.3f +/- %.3f" % (cfg, mname, r["auroc_mean"], r["auroc_std"]), flush=True)

    res = pd.DataFrame(rows)
    res["total_ms"] = res["feat_ms"] + res["pred_ms"]
    res.to_csv(HERE / "test_results.csv", index=False)

    print("\n%-6s %-5s %9s  %-15s %12s %12s %12s" %
          ("config", "model", "features", "AUROC (mean+/-sd)", "featurize ms", "predict ms", "total ms/mol"))
    for _, r in res.iterrows():
        print("%-6s %-5s %9d  %.3f +/- %.3f    %12.3f %12.3f %12.3f" %
              (r.config, r.model, r.n_features, r.auroc_mean, r.auroc_std, r.feat_ms, r.pred_ms, r.total_ms))

    best = res.loc[res.auroc_mean.idxmax()]
    # simplest/cheapest combination within one std of the best is not distinguishable from it
    tied = res[res.auroc_mean >= best.auroc_mean - best.auroc_std]
    cheap = tied.loc[tied.total_ms.idxmin()]
    print("\nbest mean AUROC : %s/%s  %.3f +/- %.3f" % (best.config, best.model, best.auroc_mean, best.auroc_std))
    print("fastest within 1 sd of best: %s/%s  %.3f +/- %.3f  (%.3f ms/mol)" %
          (cheap.config, cheap.model, cheap.auroc_mean, cheap.auroc_std, cheap.total_ms))
    print("\nNote: differences under ~0.02 AUROC are within noise for a dataset this size.")
    print("results saved to test_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="ABC", help="any of A B C D (default ABC)")
    ap.add_argument("--models", default="RF,HGB,ET,LR", type=lambda s: s.split(","))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="use the whole file, not just train+valid")
    ap.add_argument("--quick", action="store_true", help="600-molecule, 3-fold smoke test")
    args = ap.parse_args()
    self_test()
    run(args)
