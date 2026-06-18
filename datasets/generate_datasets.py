"""
generate_datasets.py

Generates local stand-in versions of the three Matbench datasets used by this
project:

    matbench_mp_gap.json.gz       -> target: band gap (eV)
    matbench_log_kvrh.json.gz     -> target: log10(bulk modulus, GPa)
    matbench_log_gvrh.json.gz     -> target: log10(shear modulus, GPa)

The real Matbench corpora are distributed through the `matminer`/Figshare
data pipeline, which is not reachable from this offline build environment.
To keep the rest of the pipeline (preprocessing, classical ML, quantum ML,
optimization, dashboard) fully runnable end-to-end without any external
network access, this script builds physically-plausible synthetic datasets
that follow the same JSON schema as the real Matbench files:

    material_id, formula, and a set of composition-derived numeric
    descriptors, plus the relevant target column.

To use the REAL Matbench data instead, install `matminer` and replace the
body of `load_or_generate()` with:

    from matminer.datasets import load_dataset
    df = load_dataset("matbench_mp_gap")

then export the resulting dataframe with `df.to_json(path, orient="records",
compression="gzip")` using the same column names produced below.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from typing import List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_MATERIALS = 300

ELEMENT_POOL: List[str] = [
    "Li", "Na", "K", "Mg", "Ca", "Al", "Si", "Ti", "Fe", "Co",
    "Ni", "Cu", "Zn", "Zr", "Nb", "Mo", "Sn", "Ba", "La", "W",
    "O", "N", "S", "Se", "Te", "P", "As", "Sb", "Bi", "C",
]


def _make_formula(rng: np.random.Generator) -> str:
    n_elems = rng.integers(2, 4)
    elems = rng.choice(ELEMENT_POOL, size=n_elems, replace=False)
    counts = rng.integers(1, 5, size=n_elems)
    parts = []
    for el, c in zip(elems, counts):
        parts.append(f"{el}{c if c > 1 else ''}")
    return "".join(parts)


def _composition_descriptors(rng: np.random.Generator, n: int) -> np.ndarray:
    """Synthetic composition-derived descriptors (mimic matminer feature columns)."""
    mean_atomic_mass = rng.uniform(10, 200, n)
    mean_electronegativity = rng.uniform(0.7, 3.98, n)
    mean_atomic_radius = rng.uniform(0.3, 2.5, n)
    valence_electrons = rng.uniform(1, 8, n)
    density = rng.uniform(1.0, 15.0, n)
    packing_fraction = rng.uniform(0.3, 0.74, n)
    formation_energy = rng.uniform(-4.0, 0.5, n)
    num_atoms_cell = rng.integers(2, 40, n).astype(float)
    space_group = rng.integers(1, 230, n).astype(float)
    avg_bond_length = rng.uniform(1.5, 3.2, n)
    return np.column_stack([
        mean_atomic_mass, mean_electronegativity, mean_atomic_radius,
        valence_electrons, density, packing_fraction, formation_energy,
        num_atoms_cell, space_group, avg_bond_length,
    ])


FEATURE_NAMES = [
    "mean_atomic_mass", "mean_electronegativity", "mean_atomic_radius",
    "valence_electrons", "density", "packing_fraction", "formation_energy",
    "num_atoms_cell", "space_group", "avg_bond_length",
]


def _save_gzip_json(records: list, path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(records, f)
    logger.info("Wrote %d records to %s", len(records), path)


def generate_mp_gap(out_dir: str) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    ids = [f"mp-{1000 + i}" for i in range(N_MATERIALS)]
    formulas = [_make_formula(rng) for _ in range(N_MATERIALS)]
    feats = _composition_descriptors(rng, N_MATERIALS)

    noise = rng.normal(0, 0.25, N_MATERIALS)
    gap = (
        0.015 * feats[:, 1] * feats[:, 3]
        - 0.4 * feats[:, 6]
        + 0.002 * feats[:, 4] * feats[:, 2]
        + noise
    )
    gap = np.clip(gap, 0.0, 9.0)

    records = []
    for i in range(N_MATERIALS):
        rec = {"material_id": ids[i], "formula": formulas[i]}
        rec.update({FEATURE_NAMES[j]: float(feats[i, j]) for j in range(len(FEATURE_NAMES))})
        rec["mp_gap"] = float(gap[i])
        records.append(rec)
    _save_gzip_json(records, os.path.join(out_dir, "matbench_mp_gap.json.gz"))


def generate_log_kvrh(out_dir: str) -> None:
    rng = np.random.default_rng(RANDOM_STATE + 1)
    ids = [f"mp-{1000 + i}" for i in range(N_MATERIALS)]
    formulas = [_make_formula(rng) for _ in range(N_MATERIALS)]
    feats = _composition_descriptors(rng, N_MATERIALS)

    noise = rng.normal(0, 0.05, N_MATERIALS)
    log_kvrh = (
        1.4
        + 0.004 * feats[:, 0]
        + 0.25 * feats[:, 5]
        - 0.05 * feats[:, 9]
        + noise
    )
    log_kvrh = np.clip(log_kvrh, 0.5, 3.0)

    records = []
    for i in range(N_MATERIALS):
        rec = {"material_id": ids[i], "formula": formulas[i]}
        rec.update({FEATURE_NAMES[j]: float(feats[i, j]) for j in range(len(FEATURE_NAMES))})
        rec["log_kvrh"] = float(log_kvrh[i])
        records.append(rec)
    _save_gzip_json(records, os.path.join(out_dir, "matbench_log_kvrh.json.gz"))


def generate_log_gvrh(out_dir: str) -> None:
    rng = np.random.default_rng(RANDOM_STATE + 2)
    ids = [f"mp-{1000 + i}" for i in range(N_MATERIALS)]
    formulas = [_make_formula(rng) for _ in range(N_MATERIALS)]
    feats = _composition_descriptors(rng, N_MATERIALS)

    noise = rng.normal(0, 0.05, N_MATERIALS)
    log_gvrh = (
        0.9
        + 0.003 * feats[:, 0]
        + 0.2 * feats[:, 5]
        - 0.04 * feats[:, 9]
        + noise
    )
    log_gvrh = np.clip(log_gvrh, 0.2, 2.5)

    records = []
    for i in range(N_MATERIALS):
        rec = {"material_id": ids[i], "formula": formulas[i]}
        rec.update({FEATURE_NAMES[j]: float(feats[i, j]) for j in range(len(FEATURE_NAMES))})
        rec["log_gvrh"] = float(log_gvrh[i])
        records.append(rec)
    _save_gzip_json(records, os.path.join(out_dir, "matbench_log_gvrh.json.gz"))


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    generate_mp_gap(out_dir)
    generate_log_kvrh(out_dir)
    generate_log_gvrh(out_dir)
    logger.info("All synthetic Matbench-style datasets generated in %s", out_dir)


if __name__ == "__main__":
    main()
