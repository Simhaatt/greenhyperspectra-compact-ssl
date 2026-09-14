from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"
BEAM_DIR = RESULTS / "beam"
TABLE_DIR = RESULTS / "paper_tables"
OUT_TABLE = TABLE_DIR / "table44_selected_wavelengths_by_trait.csv"
OUT_LONG = TABLE_DIR / "table45_selected_wavelengths_long_interpretation.csv"
OUT_MD = RESULTS / "selected_wavelength_physiological_interpretation.md"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


def assign_feature(w: int) -> tuple[str, str]:
    if 430 <= w <= 450:
        return "blue chlorophyll absorption", "chlorophyll/carotenoid absorption region"
    if 500 <= w <= 570:
        return "green pigment reflectance", "green peak and carotenoid/chlorophyll balance"
    if 630 <= w <= 690:
        return "red chlorophyll absorption", "chlorophyll absorption and pigment concentration"
    if 700 <= w <= 740:
        return "red-edge inflection", "chlorophyll-structure transition"
    if 930 <= w <= 990:
        return "970 nm water absorption", "leaf/canopy water sensitivity"
    if 1150 <= w <= 1230:
        return "1200 nm water/dry matter shoulder", "water and dry matter continuum"
    if 1400 <= w <= 1500:
        return "1450 nm water absorption", "strong water absorption"
    if 1650 <= w <= 1800:
        return "SWIR biochemical absorption", "cellulose/lignin/protein/dry matter sensitivity"
    if 1900 <= w <= 2200:
        return "SWIR water/biochemical absorption", "water, cellulose, lignin, protein and dry matter sensitivity"
    if 740 <= w <= 1300:
        return "NIR structure plateau", "leaf/canopy structure and scattering"
    return "other spectral region", "supporting or continuum wavelength"


def selected_waves(trait: str) -> list[int]:
    path = BEAM_DIR / f"{trait}_beam_band_selection_results.csv"
    df = pd.read_csv(path)
    row = df[df["budget"] == 30].sort_values("rmse").iloc[0]
    return [int(x) for x in str(row["selected_wavelengths"]).split(",") if x.strip()]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    long_rows = []
    for trait in TRAITS:
        waves = selected_waves(trait)
        rows.append(
            {
                "trait": trait,
                "n_selected": len(waves),
                "selected_wavelengths_nm": ",".join(str(w) for w in waves),
                "representative_wavelengths_nm": ",".join(str(w) for w in waves[:15]),
            }
        )
        for rank, w in enumerate(waves, start=1):
            feature, meaning = assign_feature(w)
            long_rows.append(
                {
                    "trait": trait,
                    "rank": rank,
                    "wavelength_nm": w,
                    "spectral_feature": feature,
                    "physiological_interpretation": meaning,
                }
            )

    wide = pd.DataFrame(rows)
    long = pd.DataFrame(long_rows)
    wide.to_csv(OUT_TABLE, index=False)
    long.to_csv(OUT_LONG, index=False)

    lines = [
        "# Physiological Interpretation of BeamSearch-Selected Wavelengths",
        "",
        "BeamSearch selected 30 wavelengths per trait. The exact wavelength values are listed in the output tables, and the interpretation below maps them to broad vegetation spectral features rather than treating every exact wavelength as uniquely causal.",
        "",
        "Literature anchors: Curran (1989), PROSPECT leaf optics work including Jacquemoud et al. (2009) and Feret et al. (2008), and imaging spectroscopy synthesis from Ustin and Gamon (2010).",
        "",
    ]
    for trait in TRAITS:
        sub = long[long["trait"] == trait]
        counts = sub["spectral_feature"].value_counts().head(6)
        lines.append(f"## {trait}")
        lines.append("")
        lines.append(f"Selected wavelengths: {wide[wide['trait'] == trait].iloc[0]['selected_wavelengths_nm']}")
        lines.append("")
        lines.append("Dominant interpreted regions:")
        for name, count in counts.items():
            lines.append(f"- {name}: {count} selected bands")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_TABLE)
    print(OUT_LONG)
    print(OUT_MD)


if __name__ == "__main__":
    main()
