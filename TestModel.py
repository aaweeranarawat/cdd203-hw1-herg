"""
Check your submission before you hand it in. From your repository folder, inside the
class environment, run this on your training script:

    python TestModel.py Part1_FIRST1-LAST1-FIRST2-LAST2.py
    python TestModel.py Part1_FIRST1-LAST1-FIRST2-LAST2.py "CCO"   # also score your own SMILES

Every line should start with PASS. Compare the printed predictions with your web app.
This does not re-run your training, so also check that running your training script
from scratch recreates your ONNX file.
"""

import importlib.util, sys, zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLES = ["CCO",                                                         # tiny
           "CC(=O)Oc1ccccc1C(=O)O",                                       # aspirin
           "CN(C)CCOC(c1ccccc1)c1ccccc1.Cl",                              # a salt (two fragments)
           "C[NH2+]CC[C@@H](Oc1ccc(cc1)C(F)(F)F)c1ccccc1",                # charge and stereochemistry
           "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",   # terfenadine, a hERG blocker
           "C" + "OCC" * 64 + "O"]                                        # longer than any in hw1data.tab
ALLOWED = {"numpy==1.26.4", "pandas==2.3.3", "scikit-learn==1.6.1", "onnxruntime==1.19.2",
           "skl2onnx==1.20.0", "gradio==5.35.0", "rdkit==2023.9.6", "torch==2.2.2", "transformers==4.46.3"}
MAX_MB, MAX_PARAMETERS = 50, 150e6


def check(passed, message):
    print("PASS " if passed else "FAIL ", message)


if len(sys.argv) < 2:
    sys.exit(__doc__)
script = Path(sys.argv[1]).resolve()
folder = script.parent
smiles = SAMPLES + sys.argv[2:]

sys.path.insert(0, str(folder))   # lets your script import files sitting next to it
spec = importlib.util.spec_from_file_location("Part1", script)
part1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(part1)

# Your settings
check(part1.NAME != "GROUP-MEMBER-NAMES-HERE", "NAME is filled in (%s)" % part1.NAME)
check(0 < part1.THRESHOLD < 1 and part1.THRESHOLD != 0.5,
      "THRESHOLD is between 0 and 1 and is not 0.5 (%s)" % part1.THRESHOLD)

# featurize: SMILES strings in, one row of numbers per molecule out
features = np.asarray(part1.featurize(smiles), float)
check(features.ndim == 2 and len(features) == len(smiles),
      "featurize returns a 2-D array with one row per molecule %s" % (features.shape,))
check(np.isfinite(features).all(), "features are all finite numbers (no NaN or infinity)")
torch = sys.modules.get("torch")   # only loaded if your script uses PyTorch
parameters = sum(p.numel() for value in vars(part1).values()
                 if torch and isinstance(value, torch.nn.Module) for p in value.parameters())
check(parameters <= MAX_PARAMETERS,
      "PyTorch models loaded at the top of your script have at most 150M parameters (%.1fM)" % (parameters / 1e6))

# The ONNX model
onnx_path = folder / ("Part1_%s.onnx" % part1.NAME)
if not onnx_path.exists():
    sys.exit("FAIL  %s not found next to your script - run your training script first" % onnx_path.name)
check(onnx_path.stat().st_size <= MAX_MB * 1e6,
      "%s is at most %d MB (%.1f MB)" % (onnx_path.name, MAX_MB, onnx_path.stat().st_size / 1e6))
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
check(len(session.get_inputs()) == 1, "the ONNX model has exactly one input")
inp = session.get_inputs()[0]
dtype = {"tensor(double)": np.float64, "tensor(int64)": np.int64}.get(inp.type, np.float32)


def score(rows):
    """P(blocker) for each row, read from the model's last output the same way the grader does."""
    output = np.asarray(session.run(None, {inp.name: np.asarray(rows, dtype)})[-1], float)
    return output[:, 1] if output.ndim == 2 and output.shape[1] == 2 else output.ravel()


try:
    scores = score(features)   # all molecules in one batch
    one_at_a_time = np.concatenate([score(part1.featurize([s])) for s in smiles])   # like a web app
except Exception as error:
    sys.exit("FAIL  the ONNX model could not score your features. Does featurize still match the\n"
             "      features you trained on, and can the number of molecules vary?\n%s" % error)
check(len(scores) == len(smiles), "the model returns one prediction per molecule")
check(np.all((scores >= -1e-3) & (scores <= 1 + 1e-3)),   # small tolerance for float32 rounding
      "predictions are probabilities between 0 and 1")
check(np.allclose(scores, one_at_a_time, atol=1e-4),
      "scoring molecules one at a time gives the same predictions as scoring them together")

# The rest of the repository
readme, requirements, app = folder / "README.md", folder / "requirements.txt", folder / "Part 2.zip"
check(readme.exists() and "http" in readme.read_text(), "README.md exists and links to your web app")
pins = {line.strip() for line in requirements.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")} if requirements.exists() else {"(missing)"}
check(pins <= ALLOWED, "requirements.txt lists only allowed libraries at allowed versions"
      + "".join("\n      not allowed: " + pin for pin in sorted(pins - ALLOWED)))
same_model = False
if app.exists():
    with zipfile.ZipFile(app) as archive:
        same_model = any(archive.read(name) == onnx_path.read_bytes()
                         for name in archive.namelist() if name.endswith(".onnx"))
check(same_model, "Part 2.zip contains an identical copy of %s" % onnx_path.name)

print("\nP(blocker)  flagged at %.2f  SMILES" % part1.THRESHOLD)
for molecule, probability in zip(smiles, np.clip(scores, 0, 1)):
    print("  %.3f      %-15s %s" % (probability, "yes" if probability >= part1.THRESHOLD else "no", molecule[:60]))
