from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from analysis_utils import load_and_preprocess, fit_fe_clustered


DATA_PATH = Path("Panel_Data 2014-24.csv")
FINAL_DIR = Path("final_outputs")
FINAL_DIR.mkdir(exist_ok=True)

PILLARS = ["shock", "distribution", "income", "distortion", "structural"]
WEIGHTS = {
    "shock": 0.25,
    "distribution": 0.20,
    "income": 0.20,
    "distortion": 0.15,
    "structural": 0.20,
}


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    smin, smax = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(smin) or pd.isna(smax) or smin == smax:
        return pd.Series(0.5, index=s.index)
    return ((s - smin) / (smax - smin)).clip(0, 1)


def write_score(df: pd.DataFrame, col: str, out_name: str) -> None:
    out = df[["region_id", "year", col]].copy().sort_values(["region_id", "year"])
    out.to_csv(FINAL_DIR / out_name, index=False)


def orthogonalize_if_needed(df: pd.DataFrame, cols: list[str], threshold: float = 0.85) -> tuple[pd.DataFrame, bool]:
    corr = df[cols].corr().abs()
    upper_vals = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if upper_vals.empty or upper_vals.max() <= threshold:
        return df, False

    out = df.copy()
    transformed_cols: list[str] = []
    for col in cols:
        y = out[col].astype(float).values
        if not transformed_cols:
            transformed = y
        else:
            x = out[transformed_cols].astype(float).values
            x = np.column_stack([np.ones(len(x)), x])
            beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            y_hat = x @ beta
            transformed = (y - y_hat) + np.nanmean(y)
        out[col] = minmax(pd.Series(transformed, index=out.index))
        transformed_cols.append(col)
    return out, True


def step2_revised(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().sort_values(["region_id", "year"])

    # Normalize key regressors for numerical stability
    for v in ["mgnrega", "rainfall", "population", "mgnrega_lag1"]:
        if v in work.columns:
            mu, sd = work[v].mean(skipna=True), work[v].std(skipna=True)
            if pd.notna(sd) and sd > 0:
                work[f"{v}_z"] = (work[v] - mu) / sd
            else:
                work[f"{v}_z"] = 0.0

    # Pillar 1: Shock absorption with negative shock dummy
    rain_mean = work.groupby("region_id")["rainfall"].transform("mean")
    rain_std = work.groupby("region_id")["rainfall"].transform("std")
    work["rainfall_shock_neg"] = (work["rainfall"] < (rain_mean - rain_std)).astype(int)
    work["mgnrega_x_negshock"] = work["mgnrega_z"] * work["rainfall_shock_neg"]
    p1_formula = (
        "income ~ mgnrega_z + rainfall_shock_neg + mgnrega_z:rainfall_shock_neg + "
        "rainfall_z + population_z + C(region_id) + C(year)"
    )
    p1_res, p1_df = fit_fe_clustered(p1_formula, work)
    b3 = p1_res.params.get("mgnrega_z:rainfall_shock_neg", 0.0)
    p1_df["shock"] = minmax(b3 * p1_df["mgnrega_x_negshock"])

    # Pillar 2: Distributional effect with subgroup outcomes
    p1_df["scst_income"] = p1_df["income"] * p1_df["scst_share"]
    p1_df["women_income"] = p1_df["income"] * p1_df["women_share"]
    p2a_formula = "scst_income ~ mgnrega_z + rainfall_z + population_z + C(region_id) + C(year)"
    p2b_formula = "women_income ~ mgnrega_z + rainfall_z + population_z + C(region_id) + C(year)"
    p2a_res, p2a_df = fit_fe_clustered(p2a_formula, p1_df)
    p2b_res, p2b_df = fit_fe_clustered(p2b_formula, p1_df)
    common2 = p2a_df[["region_id", "year", "mgnrega_z"]].merge(
        p2b_df[["region_id", "year", "mgnrega_z"]], on=["region_id", "year"], suffixes=("_a", "_b")
    )
    b_scst = p2a_res.params.get("mgnrega_z", 0.0)
    b_women = p2b_res.params.get("mgnrega_z", 0.0)
    common2["distribution"] = minmax(((b_scst * common2["mgnrega_z_a"]) + (b_women * common2["mgnrega_z_b"])) / 2.0)

    # Pillar 3: Income effect
    p3_formula = "income ~ mgnrega_z + rainfall_z + population_z + C(region_id) + C(year)"
    p3_res, p3_df = fit_fe_clustered(p3_formula, p1_df)
    b_income = p3_res.params.get("mgnrega_z", 0.0)
    p3_df["income_score"] = minmax(b_income * p3_df["mgnrega_z"])

    # Pillar 4: Distortion via elasticities
    p1_df["log_income"] = np.log(p1_df["income"].clip(lower=1e-6))
    p1_df["log_agri_yield"] = np.log(p1_df["agri_yield"].clip(lower=1e-6))
    p4a_formula = "log_income ~ mgnrega_z + rainfall_z + population_z + C(region_id) + C(year)"
    p4b_formula = "log_agri_yield ~ mgnrega_z + rainfall_z + population_z + C(region_id) + C(year)"
    p4a_res, p4a_df = fit_fe_clustered(p4a_formula, p1_df)
    p4b_res, p4b_df = fit_fe_clustered(p4b_formula, p1_df)
    b_log_inc = p4a_res.params.get("mgnrega_z", 0.0)
    b_log_yield = p4b_res.params.get("mgnrega_z", 0.0)
    distortion_scalar = b_log_inc / (abs(b_log_yield) + 1e-6)
    common4 = p4a_df[["region_id", "year", "mgnrega_z"]].merge(
        p4b_df[["region_id", "year"]], on=["region_id", "year"], how="inner"
    )
    distortion_raw = distortion_scalar * common4["mgnrega_z"]
    common4["distortion"] = 1 - minmax(distortion_raw)  # higher distortion -> lower welfare score

    # Pillar 5: Structural transformation (persistence)
    p5_formula = "income ~ mgnrega_lag1_z + rainfall_z + population_z + C(region_id) + C(year)"
    p5_res, p5_df = fit_fe_clustered(p5_formula, p1_df)
    b_lag = p5_res.params.get("mgnrega_lag1_z", 0.0)
    p5_df["structural"] = minmax(b_lag * p5_df["mgnrega_lag1_z"])

    # Merge all region-year pillar scores
    merged = p1_df[["region_id", "year", "shock"]].merge(
        common2[["region_id", "year", "distribution"]], on=["region_id", "year"], how="inner"
    )
    merged = merged.merge(
        p3_df[["region_id", "year", "income_score"]].rename(columns={"income_score": "income"}),
        on=["region_id", "year"],
        how="inner",
    )
    merged = merged.merge(common4[["region_id", "year", "distortion"]], on=["region_id", "year"], how="inner")
    merged = merged.merge(p5_df[["region_id", "year", "structural"]], on=["region_id", "year"], how="inner")
    merged = merged.dropna(subset=PILLARS).copy().sort_values(["region_id", "year"])

    # Critical check: remove multicollinearity if needed
    merged, adjusted = orthogonalize_if_needed(merged, PILLARS, threshold=0.85)
    merged["orthogonalization_applied"] = int(adjusted)

    # Save revised Step 2 outputs (requested in final_outputs)
    write_score(merged.rename(columns={"shock": "pillar1_shock_score"}), "pillar1_shock_score", "pillar1_shock_score.csv")
    write_score(
        merged.rename(columns={"distribution": "pillar2_distribution_score"}),
        "pillar2_distribution_score",
        "pillar2_distribution_score.csv",
    )
    write_score(merged.rename(columns={"income": "pillar3_income_score"}), "pillar3_income_score", "pillar3_income_score.csv")
    write_score(
        merged.rename(columns={"distortion": "pillar4_distortion_score"}),
        "pillar4_distortion_score",
        "pillar4_distortion_score.csv",
    )
    write_score(
        merged.rename(columns={"structural": "pillar5_structural_score"}),
        "pillar5_structural_score",
        "pillar5_structural_score.csv",
    )
    return merged


def step3_robust(pillars_df: pd.DataFrame, base_df: pd.DataFrame) -> None:
    df = pillars_df.copy()
    df["composite_equal"] = df[PILLARS].mean(axis=1)
    df["composite_weighted"] = sum(df[k] * v for k, v in WEIGHTS.items())

    composite = df[["region_id", "year"] + PILLARS + ["composite_equal", "composite_weighted"]].copy()
    composite.to_csv(FINAL_DIR / "composite_index.csv", index=False)

    time_trend = (
        composite.groupby("year", as_index=False)[PILLARS + ["composite_equal", "composite_weighted"]]
        .mean()
        .sort_values("year")
    )
    time_trend.to_csv(FINAL_DIR / "time_trend.csv", index=False)

    covid_df = composite.copy()
    covid_df["period"] = np.where(covid_df["year"].between(2015, 2019), "Pre-COVID", "Post-COVID")
    covid_comp = covid_df.groupby("period", as_index=False)[PILLARS + ["composite_equal", "composite_weighted"]].mean()
    covid_comp.to_csv(FINAL_DIR / "covid_comparison.csv", index=False)

    corr = composite[PILLARS + ["composite_equal"]].corr()
    corr.to_csv(FINAL_DIR / "correlation_matrix.csv")

    # Heterogeneity: high vs low rainfall, high vs low income (region-level medians)
    base = base_df.copy()
    base_region = base.groupby("region_id", as_index=False).agg(rainfall=("rainfall", "mean"), income=("income", "mean"))
    rain_med = base_region["rainfall"].median()
    inc_med = base_region["income"].median()
    base_region["rainfall_group"] = np.where(base_region["rainfall"] >= rain_med, "High Rainfall", "Low Rainfall")
    base_region["income_group"] = np.where(base_region["income"] >= inc_med, "High Income", "Low Income")

    het = composite.merge(base_region[["region_id", "rainfall_group", "income_group"]], on="region_id", how="left")
    het["period"] = np.where(het["year"].between(2015, 2019), "Pre-COVID", "Post-COVID")

    h1 = het.groupby(["period", "rainfall_group"], as_index=False)[["composite_equal", "composite_weighted"]].mean()
    h1["split_type"] = "rainfall_group"
    h1 = h1.rename(columns={"rainfall_group": "group"})

    h2 = het.groupby(["period", "income_group"], as_index=False)[["composite_equal", "composite_weighted"]].mean()
    h2["split_type"] = "income_group"
    h2 = h2.rename(columns={"income_group": "group"})

    heterogeneity = pd.concat([h1, h2], ignore_index=True)
    heterogeneity.to_csv(FINAL_DIR / "heterogeneity_analysis.csv", index=False)


def main():
    base = load_and_preprocess(str(DATA_PATH))
    pillars = step2_revised(base)
    step3_robust(pillars, base)
    print("Revised Step 2 + Robust Step 3 completed.")


if __name__ == "__main__":
    main()
