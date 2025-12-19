from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api import types as ptypes

@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

def summarize_dataset(df: pd.DataFrame, example_values_per_column: int = 3) -> DatasetSummary:
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = missing / n_rows if n_rows else 0.0
        unique = int(s.nunique(dropna=True))

        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = ptypes.is_numeric_dtype(s)

        min_v = max_v = mean_v = std_v = None
        if is_numeric and non_null > 0:
            min_v = float(s.min())
            max_v = float(s.max())
            mean_v = float(s.mean())
            std_v = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=str(s.dtype),
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_v,
                max=max_v,
                mean=mean_v,
                std=std_v,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)

def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    missing_count = df.isna().sum()
    missing_share = missing_count / len(df)

    return (
        pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_share": missing_share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)

def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            vc = s.value_counts(dropna=True).head(top_k)
            if vc.empty:
                continue

            share = vc / vc.sum()
            result[name] = pd.DataFrame(
                {
                    "value": vc.index.astype(str),
                    "count": vc.values,
                    "share": share.values,
                }
            )

            if len(result) >= max_columns:
                break

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    high_cardinality_threshold: float = 0.5,
) -> Dict[str, Any]:
    max_missing = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0

    constant_columns = [
        c.name for c in summary.columns if c.unique <= 1
    ]
    has_constant_columns = len(constant_columns) > 0

    high_cardinality_columns = [
        c.name
        for c in summary.columns
        if not c.is_numeric and summary.n_rows > 0 and (c.unique / summary.n_rows) >= high_cardinality_threshold
    ]
    has_high_cardinality_categoricals = len(high_cardinality_columns) > 0

    too_few_rows = summary.n_rows < 100
    too_many_columns = summary.n_cols > 100
    too_many_missing = max_missing > 0.5

    score = 1.0 - max_missing
    if too_few_rows:
        score -= 0.2
    if too_many_columns:
        score -= 0.1
    if has_constant_columns:
        score -= 0.1
    if has_high_cardinality_categoricals:
        score -= 0.1

    score = max(0.0, min(1.0, score))

    return {
        # старые
        "too_few_rows": too_few_rows,
        "too_many_columns": too_many_columns,
        "max_missing_share": max_missing,
        "too_many_missing": too_many_missing,
        
        # новые
        "has_constant_columns": has_constant_columns,
        "constant_columns": constant_columns,
        "has_high_cardinality_categoricals": has_high_cardinality_categoricals,
        "high_cardinality_columns": high_cardinality_columns,

        "quality_score": score,
    }


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": c.name,
            "dtype": c.dtype,
            "non_null": c.non_null,
            "missing": c.missing,
            "missing_share": c.missing_share,
            "unique": c.unique,
            "is_numeric": c.is_numeric,
            "min": c.min,
            "max": c.max,
            "mean": c.mean,
            "std": c.std,
        }
        for c in summary.columns
    )