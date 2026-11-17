#!/usr/bin/env python3
"""Produce the two figures printed in the article from the result tables.

Figure 1 is the Ward dendrogram written by run_proximity.py for the corporate,
full-CPC specification (results/figures/assignee_dendrogram_corporate.png).
Figure 2 is drawn here from results/tables/h4_lexical_prevalence_by_quarter.csv.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import FIGURES, TABLES, ensure_output_dirs

W, H = 1968, 1248
FONT_CANDIDATES = [
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def font(size: int, bold: bool = False):
    for regular, heavy in FONT_CANDIDATES:
        path = heavy if bold else regular
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def lexical_figure(out: Path) -> None:
    with (TABLES / "h4_lexical_prevalence_by_quarter.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    series = {"generative_ai": {}, "vision": {}}
    labels = {}
    for r in rows:
        i = int(r["period_index"])
        series[r["target"]][i] = 100 * float(r["within_period_share"])
        labels[i] = r["period_label"]
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    left, top, right, bottom = 190, 130, 1880, 1015
    d.text((W // 2, 45), "Quarterly prevalence of generative-AI and vision-related lexical indicators",
           font=font(42, True), fill="#111827", anchor="ma")
    d.line((left, bottom, right, bottom), fill="#111827", width=3)
    d.line((left, top, left, bottom), fill="#111827", width=3)
    ymax = 12
    for value in range(0, 13, 2):
        y = bottom - value / ymax * (bottom - top)
        d.line((left, y, right, y), fill="#E5E7EB", width=2)
        d.text((left - 18, y), f"{value:g}", font=font(25), fill="#374151", anchor="rm")
    d.text(((left + right) / 2, 1140), "Publication quarter", font=font(30), fill="#111827", anchor="ma")
    lab = Image.new("RGBA", (750, 70), (255, 255, 255, 0))
    ImageDraw.Draw(lab).text((375, 35), "Share of sampled family abstracts (%)", font=font(30),
                             fill="#111827", anchor="mm")
    lab = lab.rotate(90, expand=True)
    im.paste(lab, (35, 300), lab)

    def xy(i, value):
        return left + i / 31 * (right - left), bottom - value / ymax * (bottom - top)

    for i in range(0, 32, 4):
        x, _ = xy(i, 0)
        d.line((x, bottom, x, bottom + 10), fill="#111827", width=2)
        d.text((x, bottom + 18), labels[i][:4], font=font(24), fill="#374151", anchor="ma")
    colors = {"generative_ai": "#176B9C", "vision": "#E07A1F"}
    for target in ("generative_ai", "vision"):
        points = [xy(i, series[target][i]) for i in sorted(series[target])]
        d.line(points, fill=colors[target], width=5, joint="curve")
        for x, y in points:
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[target])
    break_x, _ = xy(20, 0)
    for y in range(top, bottom, 22):
        d.line((break_x, y, break_x, min(y + 12, bottom)), fill="#4B5563", width=3)
    d.line((240, 1099, 290, 1099), fill="#176B9C", width=5)
    d.text((305, 1083), "Generative AI (specific expressions)", font=font(22), fill="#374151")
    d.line((900, 1099, 950, 1099), fill="#E07A1F", width=5)
    d.text((965, 1083), "Vision/image processing", font=font(22), fill="#374151")
    d.text((1480, 1083), "- -  Break: 2023Q1", font=font(22), fill="#374151")
    im.save(out, dpi=(240, 240))


def main() -> None:
    ensure_output_dirs()
    dendrogram = FIGURES / "assignee_dendrogram_corporate.png"
    if dendrogram.exists():
        shutil.copy(dendrogram, FIGURES / "figure1_assignee_dendrogram_corporate.png")
    else:
        print("Figure 1 needs run_proximity.py on data/assignee_cpc_year.parquet; "
              "the shipped copy is results/figures/figure1_assignee_dendrogram_corporate.png")
    lexical_figure(FIGURES / "figure2_lexical_prevalence.png")
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
