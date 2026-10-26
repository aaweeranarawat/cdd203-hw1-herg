"""
Homework 1, Part 1 - hERG blockade classifier.

Rename to Part1_[GROUP_NAMES].py, fill in the four TODOs, 
then run: python Part1_[GROUP_NAMES].py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

# TODO 1: your group's names -> Part1_FIRST1-LAST1-FIRST2-LAST2.onnx
NAME = "GROUP-MEMBER-NAMES-HERE"

# TODO 2: your own threshold; justify it in README.md
THRESHOLD = 0.5


def featurize(smiles):
    """TODO 3: SMILES strings -> 2D float32 NumPy array, one row per molecule.
    How you turn a SMILES string into numbers is up to you.
    Include any custom imports needed in this file and don't forget to
    update the requirements.txt with the library and exact version,
    which will need to be pre-approved by Peter if deviating from the
    versions indicated in that file.
    """
    raise NotImplementedError


def build_model():
    """TODO 4: Return your model object here.

    If you are using sklearn, then an unfitted sklearn classifier or Pipeline,
    e.g. RandomForestClassifier(...).

    Not using sklearn? Export your own ONNX with one input of shape
    (n_molecules, n_features) where n_molecules can vary (PyTorch: dynamic_axes),
    and a last output holding P(blocker), with sigmoid/softmax inside the model.
    featurize must take SMILES strings and return one row per molecule.
    TestModel.py checks this.
    """
    raise NotImplementedError


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
    X_train, y_train = prepare(pd.concat([split["train"], split["valid"]]))
    X_test, y_test = prepare(split["test"])

    model = build_model().fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    report(y_test, probability, 0.5)
    report(y_test, probability, THRESHOLD)
    save_onnx(model, X_test[:1])


if __name__ == "__main__":
    main()
