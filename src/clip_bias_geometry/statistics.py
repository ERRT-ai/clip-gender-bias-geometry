"""Statistical models used for the factorial analysis."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def factorial_anova(results: pd.DataFrame) -> pd.DataFrame:
    """Run the study's Type-II ANOVA and attach effect-size columns."""
    formula = (
        "absolute_bias ~ "
        "C(data_setting) + C(prompt_type) + C(embedding_method) + C(job) + "
        "C(data_setting):C(prompt_type) + "
        "C(data_setting):C(embedding_method) + "
        "C(prompt_type):C(embedding_method)"
    )
    model = smf.ols(formula, data=results).fit()
    table = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
    residual_ss = table.loc[table["term"] == "Residual", "sum_sq"].iloc[0]
    table["partial_eta_squared"] = table["sum_sq"] / (table["sum_sq"] + residual_ss)
    table["variance_share_eta"] = table["sum_sq"] / table["sum_sq"].sum()
    return table
