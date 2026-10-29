"""Copies the current shared module + ONNX into Part2/ and zips it as 'Part 2.zip'.
Re-run this after every retrain so the zip always holds the submitted .onnx."""
import shutil, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONNX = "Part1_Annie-Anya-Arefin-Miles.onnx"
for name in ("herg_features.py", ONNX):
    shutil.copy2(HERE / name, HERE / "Part2" / name)
with zipfile.ZipFile(HERE / "Part 2.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted((HERE / "Part2").iterdir()):
        if f.is_file() and f.name != ".DS_Store":
            z.write(f, f.name)
print("wrote Part 2.zip:", [i.filename for i in zipfile.ZipFile(HERE / "Part 2.zip").infolist()])
