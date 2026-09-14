from __future__ import annotations

import itertools
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"
INPUT = RESULTS / "beam_stability" / "beam_stability_results.csv"
OUT = RESULTS / "paper_tables" / "table41_band_stability_analysis.csv"
DETAIL_OUT = RESULTS / "beam_stability" / "band_stability_pairwise_overlap.csv"
SUMMARY_OUT = RESULTS / "summary" / "table41_band_stability_analysis.csv"


def parse_waves(value: object) -> set[int]:
    return {int(part) for part in str(value).split(",") if part.strip()}


def assign_region(wavelength: int) -> str:
    if 400 <= wavelength <= 570:
        return "blue_green_pigments_400_570"
    if 630 <= wavelength < 690:
        return "red_chlorophyll_630_690"
    if 690 <= wavelength < 740:
        return "red_edge_690_740"
    if 740 <= wavelength < 930:
        return "nir_structure_740_930"
    if 930 <= wavelength <= 1230:
        return "water_nir_930_1230"
    if 1230 < wavelength < 1400:
        return "nir_swir_transition_1230_1400"
    if 1400 <= wavelength <= 1500:
        return "swir_water_1400_1500"
    if 1500 < wavelength <= 1800:
        return "swir_biochemical_1500_1800"
    if 1900 <= wavelength <= 2200:
        return "swir_water_biochemical_1900_2200"
    return "other"


def jaccard(a: set[object], b: set[object]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df[df["budget"] == 30].copy()
    if df.empty:
        raise RuntimeError("No BeamSearch stability rows found for budget 30.")

    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for trait, sub in df.groupby("trait"):
        runs = []
        region_counter: Counter[str] = Counter()
        for _, row in sub.sort_values("seed").iterrows():
            waves = parse_waves(row["selected_wavelengths"])
            regions = {assign_region(w) for w in waves}
            region_counter.update(assign_region(w) for w in waves)
            runs.append((int(row["seed"]), waves, regions))

        for (seed_a, waves_a, regions_a), (seed_b, waves_b, regions_b) in itertools.combinations(runs, 2):
            pair_rows.append(
                {
                    "trait": trait,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "exact_overlap_count": len(waves_a & waves_b),
                    "exact_jaccard": jaccard(waves_a, waves_b),
                    "region_overlap_count": len(regions_a & regions_b),
                    "region_jaccard": jaccard(regions_a, regions_b),
                }
            )

        trait_pairs = pd.DataFrame([row for row in pair_rows if row["trait"] == trait])
        most_common = "; ".join(f"{region}:{count}" for region, count in region_counter.most_common(5))
        summary_rows.append(
            {
                "trait": trait,
                "runs": len(runs),
                "mean_exact_band_overlap": float(trait_pairs["exact_overlap_count"].mean()),
                "mean_exact_jaccard": float(trait_pairs["exact_jaccard"].mean()),
                "mean_region_overlap": float(trait_pairs["region_overlap_count"].mean()),
                "mean_region_jaccard": float(trait_pairs["region_jaccard"].mean()),
                "most_frequent_regions": most_common,
            }
        )

    pairwise = pd.DataFrame(pair_rows)
    summary = pd.DataFrame(summary_rows).sort_values("trait")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(DETAIL_OUT, index=False)
    summary.to_csv(OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(OUT)
    print(DETAIL_OUT)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
