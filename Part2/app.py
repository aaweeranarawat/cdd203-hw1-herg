"""
hERG blocker predictor - Gradio web app (Homework 1, Part 2).

Loads the submitted ONNX model and runs the exact same featurize() as the training script
(herg_features.py is the shared module, copied unchanged next to this file).
"""

import os
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from rdkit import Chem, RDLogger

from herg_features import featurize

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "Part1_Annie-Anya-Arefin-Miles.onnx"
THRESHOLD = 0.45  # must equal THRESHOLD in Part1_Annie-Anya-Arefin-Miles.py (checked by the parity test)

_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def is_valid(smiles):
    """True when RDKit can parse a non-empty molecule from the text."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None and mol.GetNumAtoms() > 0


def predict_proba(smiles):
    """P(hERG blocker) for one SMILES string, straight from the ONNX model."""
    features = featurize([smiles.strip()])
    return float(_session.run(None, {_input_name: features})[-1][0, 1])


def predict(smiles):
    """Returns (label, probability, markdown for display). Same numbers the ONNX file gives."""
    if not is_valid(smiles):
        return None, None, "Enter a valid SMILES string first."
    p = predict_proba(smiles)
    blocker = p >= THRESHOLD
    label = "hERG blocker" if blocker else "hERG non-blocker"
    if blocker:
        advice = ("The model predicts this molecule is likely to block the hERG channel. "
                  "Confirm with a laboratory hERG assay before taking it further.")
    else:
        advice = ("The model does not predict hERG blocking. This is only one cardiac check: "
                  "it says nothing about liver toxicity, other side effects or whether the drug works.")
    text = ("### %s\n\n**Model score: %.0f%%** (flagged as a blocker at %.0f%% or higher)\n\n%s"
            % (label, 100 * p, 100 * THRESHOLD, advice))
    return label, p, text


def check_input(smiles):
    """Runs on every keystroke: shows validity and enables Predict only for a valid molecule."""
    if not smiles or not smiles.strip():
        return "", gr.update(interactive=False)
    if is_valid(smiles):
        return "Valid SMILES", gr.update(interactive=True)
    return "Not a valid SMILES string", gr.update(interactive=False)


def run(smiles):
    return predict(smiles)[2]


with gr.Blocks(title="hERG blocker predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# hERG blocker predictor\n"
                "Paste a molecule as a SMILES string to estimate whether it blocks the hERG potassium "
                "channel, a common cause of drug-induced heart rhythm problems.")
    smiles_box = gr.Textbox(label="SMILES", placeholder="e.g. CCN(CC)CCOC(=O)c1ccccc1", lines=1)
    status = gr.Markdown("")
    button = gr.Button("Predict", variant="primary", interactive=False)
    result = gr.Markdown("")
    gr.Examples(
        examples=[["CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1"],
                  ["CC(=O)Oc1ccccc1C(=O)O"]],
        inputs=smiles_box, label="Examples: terfenadine (a known hERG blocker) and aspirin")
    gr.Markdown("*A research tool for early screening, not medical advice. A prediction of "
                "\"non-blocker\" does not mean a compound is safe.*")

    smiles_box.change(check_input, smiles_box, [status, button], show_progress="hidden")
    button.click(run, smiles_box, result)
    smiles_box.submit(lambda s: run(s) if is_valid(s) else "Enter a valid SMILES string first.",
                      smiles_box, result)

if __name__ == "__main__":
    # Hosts such as Render pass the port in $PORT and need the server bound to all interfaces
    hosted = "PORT" in os.environ
    demo.launch(server_name="0.0.0.0" if hosted else None,
                server_port=int(os.environ["PORT"]) if hosted else None)
