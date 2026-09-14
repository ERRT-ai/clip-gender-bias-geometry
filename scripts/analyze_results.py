from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon

from src.clip_bias_geometry.statistics import factorial_anova

RESULT_PATH = Path("artifacts/results/tables/factorial_absolute_bias_results_by_job.csv")
OUTPUT_DIR = Path("artifacts/results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def paired_embedding_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["data_setting", "prompt_type", "job"]
    rows = []
    for method_a, method_b in combinations(sorted(results["embedding_method"].unique()), 2):
        a = results[results["embedding_method"] == method_a][keys + ["absolute_bias"]].rename(
            columns={"absolute_bias": "a"}
        )
        b = results[results["embedding_method"] == method_b][keys + ["absolute_bias"]].rename(
            columns={"absolute_bias": "b"}
        )
        merged = a.merge(b, on=keys, how="inner")
        diff = merged["b"] - merged["a"]
        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "n_matched_conditions": len(merged),
            "mean_abs_bias_a": merged["a"].mean(),
            "mean_abs_bias_b": merged["b"].mean(),
            "mean_difference_b_minus_a": diff.mean(),
            "paired_t_p_value": ttest_rel(merged["a"], merged["b"]).pvalue,
            "wilcoxon_p_value": wilcoxon(merged["a"], merged["b"]).pvalue,
        })
    return pd.DataFrame(rows)


def main() -> None:
    results = pd.read_csv(RESULT_PATH)
    anova = factorial_anova(results)
    anova.to_csv(OUTPUT_DIR / "anova_table.csv", index=False)

    paired = paired_embedding_comparisons(results)
    paired.to_csv(OUTPUT_DIR / "paired_embedding_comparisons.csv", index=False)

    ranking = (
        results.groupby("embedding_method")["absolute_bias"]
        .mean()
        .sort_values()
        .rename("mean_absolute_bias")
        .reset_index()
    )
    ranking.to_csv(OUTPUT_DIR / "embedding_method_ranking.csv", index=False)

    print("\nEmbedding-method ranking")
    print(ranking.to_string(index=False))
    print("\nPaired comparisons")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
