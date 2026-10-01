"""The targets the page reports, in reading order, and their display labels.

Targets are keyed as the page names them (formation_energy_per_atom, ...). The order runs dense
and easy first, sparse and hard last; a target present in the data but absent here is reported
after these, in name order.
"""
from __future__ import annotations

from collections.abc import Iterable

ORDER = ["formation_energy_per_atom", "energy_above_hull", "band_gap", "density",
         "mag_density", "bulk_modulus", "crystal_system", "metal_insulator",
         "magnetic_label"]

LABELS = {"formation_energy_per_atom": "formation energy / atom",
          "energy_above_hull": "energy above hull",
          "band_gap": "band gap",
          "density": "density",
          "mag_density": "magnetic density",
          "bulk_modulus": "bulk modulus",
          "crystal_system": "crystal system",
          "metal_insulator": "metal / insulator",
          "magnetic_label": "magnetic / not"}


def label(target: str) -> str:
    return LABELS.get(target, target.replace("_", " "))


def ordered(present: Iterable[str]) -> list[str]:
    """`present` in reading order, with anything unlisted appended by name."""
    present = set(present)
    return [t for t in ORDER if t in present] + sorted(present - set(ORDER))
