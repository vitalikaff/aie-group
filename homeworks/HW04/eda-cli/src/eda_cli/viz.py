from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_histograms_per_column(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 6,
    bins: int = 20,
                              ) -> List[Path]:
    out_dir = _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include="number")

    paths: List[Path] = []
    for i, col in enumerate(numeric_df.columns[:max_columns]):
        s = numeric_df[col].dropna()
        if s.empty:
            continue

        fig, ax = plt.subplots()
        ax.hist(s.values, bins=bins)
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        out_path = out_dir / f"hist_{i+1}_{col}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        paths.append(out_path)

    return paths

def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Empty dataset", ha="center", va="center")
        ax.axis("off")
    else:
        mask = df.isna()
        fig, ax = plt.subplots(figsize=(min(12, df.shape[1] * 0.4), 4))
        ax.imshow(mask, aspect="auto")
        ax.set_title("Missing values matrix")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=8)
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
        ax.axis("off")
    else:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.index)
        ax.set_title("Correlation heatmap")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_top_categories_tables(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
                              ) -> List[Path]:
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []

    for name, table in top_cats.items():
        out_path = out_dir / f"top_values_{name}.csv"
        table.to_csv(out_path, index=False)
        paths.append(out_path)

    return paths
