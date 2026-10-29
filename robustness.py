"""
Robustness study for the hERG classifier.

Random k-fold CV puts near-identical molecules on both sides of every split and so
overstates how well the model handles the held-out test set, which is "different in
composition". Everything here uses Bemis-Murcko SCAFFOLD-grouped CV: a scaffold never
appears in both train and validation folds.

    1. random vs scaffold CV for configs A / B / C (RF, ET, LR)
    2. block ablation of config C (add-one and drop-one)
    3. RF hyperparameter search under scaffold CV
    4. seed variance
    5. out-of-fold threshold curve (threshold chosen on all 8,000 molecules, not 1,000)
    6. input stress test through featurize() and the exported ONNX

    python robustness.py            # full study, writes robustness_results.txt
    python robustness.py --quick    # tiny smoke test
"""

import argparse
import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("harness", HERE / "test.py")  # not `import test` (stdlib clash)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

SEED = 42


def scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "INVALID"
    mol = h._chooser.choose(mol)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "ACYCLIC"


def rf(**kw):
    p = dict(n_estimators=500, max_depth=20, min_samples_leaf=2, max_features="sqrt",
             class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    p.update(kw)
    return RandomForestClassifier(**p)


def cv_auc(make, X, y, splits):
    aucs = []
    for tr, te in splits:
        aucs.append(roc_auc_score(y[te], make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    return np.mean(aucs), np.std(aucs)


def oof(make, X, y, splits):
    p = np.zeros(len(y))
    for tr, te in splits:
        p[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def fmt(m, s):
    return "%.3f +/- %.3f" % (m, s)


def main(args):
    smiles, y = h.load_data(False)
    if args.quick:
        keep = np.random.RandomState(SEED).choice(len(smiles), 800, replace=False)
        smiles, y = [smiles[i] for i in keep], y[keep]
    groups = pd.factorize(pd.Series([scaffold(s) for s in smiles]))[0]
    print("molecules %d | distinct scaffolds %d | largest scaffold group %d" %
          (len(y), groups.max() + 1, np.bincount(groups).max()))
    blocks = h.compute_blocks(smiles)
    k = 3 if args.quick else 5
    rand = list(StratifiedKFold(k, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    scaf = list(StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, groups))
    print("scaffold folds: train/val overlap in scaffolds = %d" %
          sum(len(set(groups[tr]) & set(groups[te])) for tr, te in scaf))

    # 1 -------------------------------------------------------------------
    print("\n[1] random vs scaffold CV (AUROC, %d folds)" % k)
    print("%-7s %-5s %-16s %-16s %s" % ("config", "model", "random", "scaffold", "drop"))
    makers = {"RF": rf, "ET": lambda: ExtraTreesClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2,
                                                          class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
              "LR": lambda: h.make_models()["LR"]}
    for cfg in "ABC":
        X = h.assemble(cfg, blocks)
        for name, mk in makers.items():
            r, s = cv_auc(mk, X, y, rand), cv_auc(mk, X, y, scaf)
            print("%-7s %-5s %-16s %-16s %+.3f" % (cfg, name, fmt(*r), fmt(*s), s[0] - r[0]), flush=True)

    # 2 -------------------------------------------------------------------
    print("\n[2] block ablation, RF, scaffold CV")
    allb = ["ecfp", "desc", "fcfp", "ap", "pharm"]
    def run(bs):
        X = np.hstack([blocks[b] for b in bs]); return cv_auc(rf, X, y, scaf)
    base = run(allb)
    print("%-26s %s" % ("all five blocks (C)", fmt(*base)), flush=True)
    for b in allb:
        m = run([x for x in allb if x != b])
        print("%-26s %s   (%+.3f)" % ("drop " + b, fmt(*m), m[0] - base[0]), flush=True)
    for b in ("desc", "fcfp", "ap", "pharm"):
        m = run([b])
        print("%-26s %s" % (b + " alone", fmt(*m)), flush=True)

    # 3 -------------------------------------------------------------------
    print("\n[3] RF hyperparameters, config C, scaffold CV")
    X = h.assemble("C", blocks)
    grid = list(itertools.product([20, None], [1, 2, 5], ["sqrt", 0.05]))
    if args.quick:
        grid = grid[:2]
    res = []
    for depth, leaf, mf in grid:
        m = cv_auc(lambda: rf(max_depth=depth, min_samples_leaf=leaf, max_features=mf), X, y, scaf)
        res.append((m[0], m[1], depth, leaf, mf))
        print("depth %-5s leaf %d max_features %-5s %s" % (depth, leaf, mf, fmt(*m)), flush=True)
    best = max(res)
    print("best: depth %s leaf %s max_features %s -> %.3f" % (best[2], best[3], best[4], best[0]))

    # 4 -------------------------------------------------------------------
    print("\n[4] seed variance, RF defaults, scaffold CV")
    ms = [cv_auc(lambda s=s: rf(random_state=s), X, y, scaf)[0] for s in range(3 if args.quick else 5)]
    print("AUROC per seed:", " ".join("%.4f" % m for m in ms), "| spread %.4f" % (max(ms) - min(ms)))

    # 5 -------------------------------------------------------------------
    print("\n[5] out-of-fold threshold curve, RF defaults, config C")
    for label, splits in (("random", rand), ("scaffold", scaf)):
        p = oof(rf, X, y, splits)
        print("%s CV  pooled OOF AUROC %.3f" % (label, roc_auc_score(y, p)))
        print("  thr    acc   prec   sens   spec")
        for t in np.arange(0.30, 0.61, 0.05):
            pr = p >= t
            tp, fp = (pr & (y == 1)).sum(), (pr & (y == 0)).sum()
            fn, tn = (~pr & (y == 1)).sum(), (~pr & (y == 0)).sum()
            print("  %.2f  %.3f  %.3f  %.3f  %.3f" % (t, (tp + tn) / len(y), tp / max(tp + fp, 1), tp / (tp + fn), tn / (tn + fp)))

    # 6 -------------------------------------------------------------------
    print("\n[6] input stress test")
    from herg_features import FEATURE_LEN, featurize
    nasty = {
        "empty string": "", "whitespace": "   ", "None": None, "number": 5, "garbage": "not_a_molecule",
        "unclosed ring": "C1CC", "200 C": "C" * 200, "400 C": "C" * 400,
        "PEG 250 chars": "C" + "OCC" * 83 + "O", "macrocycle (18-crown-6)": "C1COCCOCCOCCOCCOCCO1",
        "salt 3 fragments": "CC[NH+](CC)CC.[Cl-].[Na+]", "metal complex": "[Pt](Cl)(Cl)(N)N",
        "all charged": "[NH3+]CCC[NH3+]", "isotope": "[13CH3][15NH2]", "radical": "[CH3]",
        "inorganic": "O", "single atom": "[Na+]", "stereo + aromatic": "C[C@H](N)c1ccccc1",
        "terfenadine": "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
        "unicode": "CéCO", "very long aromatic": "c1ccccc1" + "-c1ccccc1" * 30,
    }
    ok = True
    onnx_path = HERE / "Part1_Annie-Anya-Arefin-Miles.onnx"
    sess = None
    if onnx_path.exists():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for name, smi in nasty.items():
        try:
            f = featurize([smi])
            good = f.shape == (1, FEATURE_LEN) and f.dtype == np.float32 and np.isfinite(f).all()
            p = None
            if sess is not None:
                p = float(sess.run(None, {sess.get_inputs()[0].name: f})[-1][0, 1])
                good = good and 0 <= p <= 1
            print("%-5s %-26s %s" % ("PASS" if good else "FAIL", name, "" if p is None else "P=%.3f" % p))
            ok &= good
        except Exception as e:
            ok = False
            print("FAIL  %-26s raised %s: %s" % (name, type(e).__name__, e))
    print("stress test:", "ALL PASS" if ok else "FAILURES")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    t0 = time.time()
    main(ap.parse_args())
    print("\nelapsed %.0fs" % (time.time() - t0))
