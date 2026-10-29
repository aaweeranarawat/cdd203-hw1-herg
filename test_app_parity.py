"""
Parity test: the web app must give the same probabilities as the submitted ONNX file.

    python test_app_parity.py

Runs a list of SMILES through (1) the root ONNX file with the root featurize() and
(2) the app's predict_proba(), and asserts they match. Also checks that Part2/ and
'Part 2.zip' hold byte-identical copies of the ONNX model and the featurizer, that the
app's THRESHOLD equals the training script's, and that the Predict button logic is right.
"""
import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

HERE = Path(__file__).resolve().parent
ONNX = "Part1_Annie-Anya-Arefin-Miles.onnx"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.path.pop(0)
    return m


# reference: root ONNX + root featurizer
root_feats = load(HERE / "herg_features.py", "root_feats")
part1 = load(HERE / "Part1_Annie-Anya-Arefin-Miles.py", "part1")
ref = ort.InferenceSession(str(HERE / ONNX), providers=["CPUExecutionProvider"])


def reference(smiles):
    x = root_feats.featurize([smiles])
    return float(ref.run(None, {ref.get_inputs()[0].name: x})[-1][0, 1])


# the app under test (imports Part2/herg_features.py and Part2/<onnx>)
sys.modules.pop("herg_features", None)
app = load(HERE / "Part2" / "app.py", "app")

df = pd.read_csv(HERE / "data" / "hw1data.tab", sep="\t")
smiles = df.iloc[:, 1].sample(300, random_state=0).tolist()
smiles += ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN(C)CCOC(c1ccccc1)c1ccccc1.Cl",
           "C[NH2+]CC[C@@H](Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
           "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
           "C" + "OCC" * 64 + "O", "C" * 200, "C1COCCOCCOCCOCCOCCO1", "[Pt](Cl)(Cl)(N)N", "  CCO  "]

a = np.array([app.predict_proba(s) for s in smiles])
b = np.array([reference(s.strip()) for s in smiles])
assert np.allclose(a, b, atol=1e-6), "app and ONNX disagree: max diff %g" % np.abs(a - b).max()
print("PASS  app probabilities match the ONNX file on %d SMILES (max diff %.2e)" % (len(smiles), np.abs(a - b).max()))

for s in smiles[-10:]:
    label, p, _ = app.predict(s)
    assert (label == "hERG blocker") == (p >= part1.THRESHOLD)
print("PASS  app class labels use the training script's threshold")

assert app.THRESHOLD == part1.THRESHOLD
print("PASS  app THRESHOLD == Part1 THRESHOLD (%.2f)" % app.THRESHOLD)

for name in (ONNX, "herg_features.py"):
    assert sha((HERE / name).read_bytes()) == sha((HERE / "Part2" / name).read_bytes()), name + " differs in Part2/"
print("PASS  Part2/ holds identical copies of the ONNX model and herg_features.py")

zpath = HERE / "Part 2.zip"
assert zpath.exists(), "Part 2.zip missing - run build_part2.py"
with zipfile.ZipFile(zpath) as z:
    for name in (ONNX, "herg_features.py"):
        assert sha(z.read(name)) == sha((HERE / name).read_bytes()), name + " in Part 2.zip is stale"
    assert "app.py" in z.namelist() and "requirements.txt" in z.namelist()
print("PASS  Part 2.zip contains app code and an identical ONNX copy")

# Predict button must be disabled until the SMILES is valid
for text, expect in [("", False), ("   ", False), ("not_a_molecule", False), ("C1CC", False),
                     ("CCO", True), ("c1ccccc1", True)]:
    _, upd = app.check_input(text)
    assert upd["interactive"] is expect, (text, upd)
    assert app.is_valid(text) is expect
print("PASS  Predict button is disabled for empty / invalid input and enabled for valid SMILES")
print("\nALL PARITY CHECKS PASSED")
