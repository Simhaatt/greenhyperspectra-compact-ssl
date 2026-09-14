from __future__ import annotations
import os

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
SUMMARY_PATH = ROOT / "results" / "summary" / "all8_final_trait_summary.csv"
OUT_FIG = ROOT / "results" / "paper_figures" / "fig1_selected_wavelengths_annotated.png"
OUT_TABLE = ROOT / "results" / "paper_tables" / "table8_wavelength_region_counts.csv"


REGIONS = [
    ("Blue/green pigments", 400, 570, "#BFD7FF"),
    ("Red chlorophyll", 630, 690, "#FFC7C7"),
    ("Red edge", 690, 740, "#FFE29A"),
    ("NIR structure", 740, 1300, "#CFECCF"),
    ("Water 970/1200", 930, 1230, "#9FD5FF"),
    ("SWIR water 1450", 1400, 1500, "#CDB4DB"),
    ("SWIR biochem.", 1500, 1800, "#FFD6A5"),
    ("SWIR water/biochem.", 1900, 2200, "#D0D0D0"),
]


def parse_waves(value: str) -> list[int]:
    return [int(part) for part in str(value).split(",") if part.strip()]


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


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    order = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
    summary["target"] = pd.Categorical(summary["target"], categories=order, ordered=True)
    summary = summary.sort_values("target")

    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        target = str(row["target"])
        for wave in parse_waves(row["selected_wavelengths_k30"]):
            rows.append({"target": target, "wavelength_nm": wave, "region": assign_region(wave)})
    selected = pd.DataFrame(rows)
    counts = (
        selected.groupby(["target", "region"], observed=False)
        .size()
        .reset_index(name="selected_band_count")
        .sort_values(["target", "region"])
    )
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT_TABLE, index=False)

    fig, ax = plt.subplots(figsize=(13, 6.2))
    y_lookup = {target: idx for idx, target in enumerate(order)}

    for label, start, end, color in REGIONS:
        ax.axvspan(start, end, color=color, alpha=0.24, linewidth=0)

    for target in order:
        waves = selected[selected["target"] == target]["wavelength_nm"].tolist()
        y = y_lookup[target]
        ax.scatter(waves, [y] * len(waves), s=42, color="#1F4E79", edgecolor="white", linewidth=0.6, zorder=3)

    label_positions = [
        ("pigments", 485, 7.72),
        ("red Chl", 660, 7.46),
        ("red edge", 715, 7.72),
        ("NIR structure", 835, 7.72),
        ("water\n970/1200", 1080, 7.72),
        ("water\n1450", 1450, 7.72),
        ("SWIR biochem.\n1500-1800", 1650, 7.72),
        ("SWIR water/\nbiochem.", 2050, 7.72),
    ]
    for text, x, y in label_positions:
        ax.text(x, y, text, ha="center", va="bottom", fontsize=8.5, color="#2B2B2B")

    for x, text in [(680, "chlorophyll a"), (970, "water"), (1200, "water/dry matter"), (1450, "water"), (2100, "lignin/cellulose/protein")]:
        ax.axvline(x, color="#555555", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.text(x + 8, -0.68, text, rotation=90, va="bottom", ha="left", fontsize=8, color="#444444")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(390, 2225)
    ax.set_ylim(-0.85, len(order) + 0.18)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Trait")
    ax.set_title("BeamSearch K=30 selected wavelengths mapped to vegetation absorption regions")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=260)
    plt.close(fig)
    print(OUT_FIG)
    print(OUT_TABLE)


if __name__ == "__main__":
    main()
