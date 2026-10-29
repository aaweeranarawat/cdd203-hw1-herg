"""
Homework 1, Part 1 - hERG blockade classifier.

Run: python Part1_Annie-Anya-Arefin-Miles.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from herg_features import featurize  # shared with the web app so features never drift

SEED = 42
np.random.seed(SEED)

# TODO 1: your group's names -> Part1_FIRST1-LAST1-FIRST2-LAST2.onnx
NAME = "Annie-Anya-Arefin-Miles"

# TODO 2: chosen on the validation set (see report_threshold_candidates below).
# Below 0.5 on purpose: a missed blocker costs far more than a false alarm.
THRESHOLD = 0.45


def build_model():
    """TODO 4: random forest, best of the CV comparison (config C: AUROC 0.908 random CV,
    0.865 scaffold-grouped CV). max_features=0.05 beat "sqrt" by ~0.006 on every scaffold fold set."""
    return RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2,
                                  max_features=0.05, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)


def save_onnx(model, sample):
    """
    Export a fitted model. `sample` is one row of your features,
    which fixes the input type.
    """
    Path("Part1_%s.onnx" % NAME).write_bytes(
        to_onnx(model, sample, options={"zipmap": False}, target_opset=12).SerializeToString())
    print("\nwrote Part1_%s.onnx" % NAME)


def load_split():
    """
    Load in assignment data and split into train/val/test sets
    You do not need to use this exact split, as you will be evaluated
    on a held-out test set. You may use the data in any way you wish.
    """
    df = pd.read_csv("data/hw1data.tab", sep="\t")
    df.columns = ["Drug_ID", "Drug", "Y"]
    test = df.sample(frac=0.2, random_state=42)
    rest = df.drop(test.index)
    valid = rest.sample(frac=0.125, random_state=1)
    return {"train": rest.drop(valid.index), "valid": valid, "test": test}


def prepare(frame):
    """Hand SMILES strings and labels to your featurizer."""
    return np.asarray(featurize(frame["Drug"].tolist()), np.float32), frame["Y"].to_numpy()


def report(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print("\nthreshold %.2f" % threshold)
    for name, value in [("accuracy", accuracy_score(y, pred)),
                        ("sensitivity/recall", recall_score(y, pred)),
                        ("specificity", tn / (tn + fp)),
                        ("precision", precision_score(y, pred)),
                        ("f1", f1_score(y, pred)),
                        ("ROC-AUC", roc_auc_score(y, probability)),
                        ("PR-AUC", average_precision_score(y, probability))]:
        print("  %-20s%.3f" % (name, value))


def report_threshold_candidates(y, probability):
    """Threshold selection on the VALIDATION set only (the test set is never used for this)."""
    fpr, tpr, thr = roc_curve(y, probability)
    spec = 1 - fpr
    ok = spec > 0.7
    candidates = {"sensitivity >= 0.90": thr[np.argmax(tpr >= 0.90)],
                  "Youden's J": thr[np.argmax(tpr - fpr)],
                  "max sensitivity, specificity > 0.7": thr[ok][np.argmax(tpr[ok])],
                  "chosen THRESHOLD": THRESHOLD}
    print("\nthreshold candidates on the validation set (model fit on train only)")
    for label, t in candidates.items():
        pred = probability >= t
        print("  %-36s thr %.3f  sensitivity %.3f  specificity %.3f  precision %.3f" % (
            label, t, pred[y == 1].mean(), 1 - pred[y == 0].mean(), y[pred].mean()))


def main():
    """
    You do not have to use the train/test split provided to you.
    This is just an example of splitting the data.
    You may use the entire hw1data.tab as you wish.
    The dataset used to evaluate your model is not provided to you (just
    like in the real world - you train on some data then deploy your model,
    which will receive new unseen data).
    """
    split = load_split()
    X_fit, y_fit = prepare(split["train"])
    X_valid, y_valid = prepare(split["valid"])
    X_test, y_test = prepare(split["test"])

    # threshold selection: model trained on train only, scored on validation
    report_threshold_candidates(y_valid, build_model().fit(X_fit, y_fit).predict_proba(X_valid)[:, 1])

    # final model: train + validation, evaluated once on the untouched test split
    model = build_model().fit(np.vstack([X_fit, X_valid]), np.concatenate([y_fit, y_valid]))
    probability = model.predict_proba(X_test)[:, 1]
    report(y_test, probability, 0.5)
    report(y_test, probability, THRESHOLD)
    save_onnx(model, X_test[:1])


if __name__ == "__main__":
    main()
