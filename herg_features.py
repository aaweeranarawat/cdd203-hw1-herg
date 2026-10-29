"""
Shared featurizer for the hERG classifier (config C from the comparison harness).

Imported by Part1_[GROUP_NAMES].py (training) and by the Gradio app, so training and
serving always compute exactly the same features.

Layout of the feature vector (FEATURE_LEN = 4127):
    2048  Morgan ECFP4 bits (radius 2)
      18  descriptors mapped to the hERG pharmacophore
    1024  FCFP4 bits (feature-based Morgan)
    1024  atom-pair bits
      13  basic-N -> aromatic-atom bond-distance histogram (10 bins) + min / mean / max
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

N_ECFP, N_DESC, N_FCFP, N_AP, N_PHARM = 2048, 18, 1024, 1024, 13
FEATURE_LEN = N_ECFP + N_DESC + N_FCFP + N_AP + N_PHARM

# sp3 N, excluding amides, nitro-like N=*, nitriles and already-charged N
BASIC_N = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N[#6]=[O,N,S]);!$(N=*);!$(N#*);!$([N+])]")
TERTIARY_N = Chem.MolFromSmarts("[NX3;H0;!$(N[#6]=[O,N,S]);!$(N=*);!$(N#*);!$([N+])]")

_chooser = rdMolStandardize.LargestFragmentChooser()
_ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_ECFP)
_fcfp = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=N_FCFP, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
_ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_AP)


def _descriptors(mol):
    """18 descriptors, one group per known hERG blocker characteristic."""
    try:
        AllChem.ComputeGasteigerCharges(mol)
        q = np.array([float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()])
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        if q.size == 0:
            q = np.zeros(1)
    except Exception:
        q = np.zeros(1)
    n_heavy = max(mol.GetNumHeavyAtoms(), 1)
    return [
        len(mol.GetSubstructMatches(BASIC_N)),                       # basic N: cation-pi with Tyr-652
        len(mol.GetSubstructMatches(TERTIARY_N)),
        q.max(), q.min(),                                            # Gasteiger partial charges
        sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        Descriptors.MolLogP(mol), Descriptors.MolWt(mol),            # lipophilic and bulky
        rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcLabuteASA(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumAromaticRings(mol),                  # pi-pi stacking
        sum(a.GetIsAromatic() for a in mol.GetAtoms()) / n_heavy,
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),                 # fit to the inner cavity
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
    ]


def _pharmacophore(mol):
    """Bond distances from each basic N to each aromatic atom: 10-bin histogram + min/mean/max."""
    n_idx = [m[0] for m in mol.GetSubstructMatches(BASIC_N)]
    aro_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not n_idx or not aro_idx:
        return np.zeros(N_PHARM)
    dm = Chem.GetDistanceMatrix(mol)
    d = np.array([dm[i][j] for i in n_idx for j in aro_idx])
    hist, _ = np.histogram(d, bins=np.arange(0.5, 11.5, 1.0))
    return np.concatenate([hist / len(d), [d.min(), d.mean(), d.max()]])


def _one(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:  # unparseable or empty ("") -> zeros, never crash
        return np.zeros(FEATURE_LEN, np.float32)
    mol = _chooser.choose(mol)  # drop salts / counter-ions
    row = np.concatenate([_ecfp.GetFingerprintAsNumPy(mol), np.asarray(_descriptors(mol), np.float64),
                          _fcfp.GetFingerprintAsNumPy(mol), _ap.GetFingerprintAsNumPy(mol),
                          _pharmacophore(mol)])
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def featurize(smiles):
    """SMILES strings -> float32 array of shape (n_molecules, FEATURE_LEN), any SMILES length."""
    if isinstance(smiles, str):
        smiles = [smiles]
    return np.vstack([_one(s) for s in smiles]).astype(np.float32)


if __name__ == "__main__":
    assert featurize(["C" * 200]).shape[1] == FEATURE_LEN
    assert featurize(["not_a_molecule"]).shape[1] == FEATURE_LEN
    assert featurize(["CCO"]).shape[1] == FEATURE_LEN
    assert not featurize(["not_a_molecule"]).any()
    assert featurize(["CCO"]).dtype == np.float32
    print("featurize OK, FEATURE_LEN =", FEATURE_LEN)
