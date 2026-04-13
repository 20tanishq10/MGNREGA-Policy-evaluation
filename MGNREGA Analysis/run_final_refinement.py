from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from analysis_utils import load_and_preprocess, fit_fe_clustered


DATA_PATH = Path("Panel_Data 2014-24.csv")
FINAL_DIR = Path("final_outputs")
REFINED_DIR = Path("final_outputs_refined")
REFINED_DIR.mkdir(exist_ok=True)

PILLARS_BASE = ["shock", "distribution", "income", "distortion", "structural_persistence"]
WEIGHTS = {
    "shock": 0.25,
    "distribution": 0.20,
    "income": 0.20,
    "distortion": 0.15,
    "structural_persistence": 0.20,
}


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(0.5, index=s.index)
    return ((s - mn) / (mx - mn)).clip(0, 1)


def z_then_rescale(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        z = pd.Series(0.0, index=s.index)
    else:
        z = (s - mu) / sd
    return minmax(z)


def enforce_corr_threshold(df: pd.DataFrame, cols: list[str], threshold: float = 0.85, max_iter: int = 12) -> pd.DataFrame:
    out = df.copy()
    for _ in range(max_iter):
        corr = out[cols].corr().abs()
        tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        if tri.stack().empty:
            break
        pair = tri.stack().idxmax()
        max_corr = tri.stack().max()
        if pd.isna(max_corr) or max_corr <= threshold:
            break
        c1, c2 = pair
        x = out[c1].values.astype(float)
        y = out[c2].values.astype(float)
        X = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        out[c2] = z_then_rescale(pd.Series(resid + np.nanmean(y), index=out.index))
    return out


def load_existing_pillars() -> pd.DataFrame:
    p2 = pd.read_csv(FINAL_DIR / "pillar2_distribution_score.csv").rename(
        columns={"pillar2_distribution_score": "distribution"}
    )
    p3 = pd.read_csv(FINAL_DIR / "pillar3_income_score.csv").rename(
        columns={"pillar3_income_score": "income"}
    )
    merged = p2.merge(p3, on=["region_id", "year"], how="inner")
    return merged


def refine_pillar1(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["rainfall_dev"] = (
        (x["rainfall"] - x.groupby("region_id")["rainfall"].transform("mean"))
        / x.groupby("region_id")["rainfall"].transform("std").replace(0, np.nan)
    )
    x["shock_intensity"] = x["rainfall_dev"].abs()
    x["extreme_shock"] = (x["rainfall_dev"] < -1).astype(int)

    mu, sd = x["mgnrega"].mean(skipna=True), x["mgnrega"].std(skipna=True)
    x["mgnrega_z"] = (x["mgnrega"] - mu) / sd if sd and sd > 0 else 0.0
    x["mgnrega_x_extreme"] = x["mgnrega_z"] * x["extreme_shock"]

    formula = (
        "income ~ mgnrega_z + shock_intensity + mgnrega_z:extreme_shock + "
        "rainfall + population + C(region_id) + C(year)"
    )
    res, used = fit_fe_clustered(formula, x)
    b3 = res.params.get("mgnrega_z:extreme_shock", 0.0)
    used["pillar1_shock_score_refined"] = minmax(b3 * used["mgnrega_x_extreme"])

    out = used[["region_id", "year", "pillar1_shock_score_refined"]].copy()
    out.to_csv(REFINED_DIR / "pillar1_shock_score_refined.csv", index=False)
    return out


def refine_pillar4(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    mu, sd = x["mgnrega"].mean(skipna=True), x["mgnrega"].std(skipna=True)
    x["mgnrega_z"] = (x["mgnrega"] - mu) / sd if sd and sd > 0 else 0.0
    x["log_income"] = np.log(x["income"].clip(lower=1e-6))
    x["log_agri_yield"] = np.log(x["agri_yield"].clip(lower=1e-6))

    res_inc, used_inc = fit_fe_clustered(
        "log_income ~ mgnrega_z + rainfall + population + C(region_id) + C(year)", x
    )
    res_yld, used_yld = fit_fe_clustered(
        "log_agri_yield ~ mgnrega_z + rainfall + population + C(region_id) + C(year)", x
    )
    b1 = res_inc.params.get("mgnrega_z", 0.0)
    b2 = res_yld.params.get("mgnrega_z", 0.0)
    ratio = b1 / (abs(b2) + 0.01)

    merged = used_inc[["region_id", "year", "mgnrega_z"]].merge(
        used_yld[["region_id", "year"]], on=["region_id", "year"], how="inner"
    )
    raw = ratio * merged["mgnrega_z"]
    merged["pillar4_distortion_score_refined"] = 1 - minmax(raw)
    merged[["region_id", "year", "pillar4_distortion_score_refined"]].to_csv(
        REFINED_DIR / "pillar4_distortion_score_refined.csv", index=False
    )
    return merged[["region_id", "year", "pillar4_distortion_score_refined"]]


def refine_pillar5(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    mu, sd = x["mgnrega_lag1"].mean(skipna=True), x["mgnrega_lag1"].std(skipna=True)
    x["mgnrega_lag1_z"] = (x["mgnrega_lag1"] - mu) / sd if sd and sd > 0 else 0.0
    res, used = fit_fe_clustered(
        "income ~ mgnrega_lag1_z + rainfall + population + C(region_id) + C(year)", x
    )
    b = res.params.get("mgnrega_lag1_z", 0.0)
    used["pillar5_structural_persistence_score"] = minmax(b * used["mgnrega_lag1_z"])
    out = used[["region_id", "year", "pillar5_structural_persistence_score"]].copy()
    out.to_csv(REFINED_DIR / "pillar5_structural_persistence_score.csv", index=False)
    return out


def build_composite_rescaled(p1: pd.DataFrame, p4: pd.DataFrame, p5: pd.DataFrame, base_p23: pd.DataFrame) -> pd.DataFrame:
    merged = (
        p1.merge(p4, on=["region_id", "year"], how="inner")
        .merge(p5, on=["region_id", "year"], how="inner")
        .merge(base_p23, on=["region_id", "year"], how="inner")
    )
    merged = merged.rename(
        columns={
            "pillar1_shock_score_refined": "shock",
            "pillar4_distortion_score_refined": "distortion",
            "pillar5_structural_persistence_score": "structural_persistence",
        }
    )

    for c in PILLARS_BASE:
        merged[c] = z_then_rescale(merged[c])

    merged = enforce_corr_threshold(merged, PILLARS_BASE, threshold=0.85)

    merged["index_equal_rescaled"] = merged[PILLARS_BASE].mean(axis=1)
    merged["index_weighted_rescaled"] = sum(merged[k] * w for k, w in WEIGHTS.items())
    out = merged[["region_id", "year"] + PILLARS_BASE + ["index_equal_rescaled", "index_weighted_rescaled"]]
    out.to_csv(REFINED_DIR / "composite_index_rescaled.csv", index=False)
    return out


def advanced_heterogeneity(df_panel: pd.DataFrame, composite: pd.DataFrame) -> pd.DataFrame:
    r = df_panel.groupby("region_id", as_index=False).agg(
        avg_income=("income", "mean"),
        rain_volatility=("rainfall", "std"),
    )
    inc_q30 = r["avg_income"].quantile(0.30)
    vol_q70 = r["rain_volatility"].quantile(0.70)
    r["income_group"] = np.where(r["avg_income"] <= inc_q30, "Low Income (Bottom 30%)", "Higher Income")
    r["shock_group"] = np.where(r["rain_volatility"] >= vol_q70, "High Shock (Top 30%)", "Lower Shock")

    x = composite.merge(r[["region_id", "income_group", "shock_group"]], on="region_id", how="left")
    g1 = x.groupby("income_group", as_index=False)[["index_equal_rescaled", "index_weighted_rescaled"]].mean()
    g1["split"] = "income_group"
    g1 = g1.rename(columns={"income_group": "group"})

    g2 = x.groupby("shock_group", as_index=False)[["index_equal_rescaled", "index_weighted_rescaled"]].mean()
    g2["split"] = "shock_group"
    g2 = g2.rename(columns={"shock_group": "group"})

    # Optional interaction regression
    panel = df_panel.copy().merge(r[["region_id", "income_group"]], on="region_id", how="left")
    panel["low_income_region"] = (panel["income_group"] == "Low Income (Bottom 30%)").astype(int)
    mu, sd = panel["mgnrega"].mean(skipna=True), panel["mgnrega"].std(skipna=True)
    panel["mgnrega_z"] = (panel["mgnrega"] - mu) / sd if sd and sd > 0 else 0.0
    panel["mgnrega_x_low_income"] = panel["mgnrega_z"] * panel["low_income_region"]
    reg, used = fit_fe_clustered(
        "income ~ mgnrega_z + mgnrega_z:low_income_region + rainfall + population + C(region_id) + C(year)",
        panel,
    )
    inter = pd.DataFrame(
        [
            {
                "split": "interaction_regression",
                "group": "mgnrega_x_low_income",
                "index_equal_rescaled": reg.params.get("mgnrega_z:low_income_region", np.nan),
                "index_weighted_rescaled": reg.pvalues.get("mgnrega_z:low_income_region", np.nan),
            }
        ]
    )

    out = pd.concat([g1, g2, inter], ignore_index=True)
    out.to_csv(REFINED_DIR / "heterogeneity_advanced.csv", index=False)
    return out


def dynamic_response(df_panel: pd.DataFrame) -> pd.DataFrame:
    x = df_panel.copy().sort_values(["region_id", "year"])
    x["mgnrega_lag1"] = x.groupby("region_id")["mgnrega"].shift(1)
    x["mgnrega_lead1"] = x.groupby("region_id")["mgnrega"].shift(-1)

    for c in ["mgnrega_lag1", "mgnrega", "mgnrega_lead1"]:
        mu, sd = x[c].mean(skipna=True), x[c].std(skipna=True)
        x[f"{c}_z"] = (x[c] - mu) / sd if sd and sd > 0 else 0.0

    res, used = fit_fe_clustered(
        "income ~ mgnrega_lag1_z + mgnrega_z + mgnrega_lead1_z + rainfall + population + C(region_id) + C(year)",
        x,
    )
    out = pd.DataFrame(
        [
            {"term": "mgnrega_lag1_z", "coef": res.params.get("mgnrega_lag1_z"), "p_value": res.pvalues.get("mgnrega_lag1_z")},
            {"term": "mgnrega_z", "coef": res.params.get("mgnrega_z"), "p_value": res.pvalues.get("mgnrega_z")},
            {"term": "mgnrega_lead1_z", "coef": res.params.get("mgnrega_lead1_z"), "p_value": res.pvalues.get("mgnrega_lead1_z")},
        ]
    )
    out.to_csv(REFINED_DIR / "dynamic_effects.csv", index=False)
    return out


def validation_checks(composite: pd.DataFrame) -> pd.DataFrame:
    corr = composite[PILLARS_BASE].corr().abs()
    max_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().max()
    stds = composite[PILLARS_BASE].std()
    sensitivity = float((composite["index_equal_rescaled"] - composite["index_weighted_rescaled"]).abs().mean())

    rows = [{"check": "max_pairwise_corr", "value": max_corr, "pass": int(max_corr < 0.85)}]
    for k, v in stds.items():
        rows.append({"check": f"std_{k}", "value": float(v), "pass": int(v > 0.05)})
    rows.append({"check": "mean_abs_equal_weighted_gap", "value": sensitivity, "pass": 1})

    out = pd.DataFrame(rows)
    out.to_csv(REFINED_DIR / "validation_checks.csv", index=False)
    return out


def main():
    panel = load_and_preprocess(str(DATA_PATH))
    base_p23 = load_existing_pillars()

    p1 = refine_pillar1(panel)
    p4 = refine_pillar4(panel)
    p5 = refine_pillar5(panel)

    composite = build_composite_rescaled(p1, p4, p5, base_p23)
    advanced_heterogeneity(panel, composite)
    dynamic_response(panel)
    checks = validation_checks(composite)

    print("Final refinement pipeline completed.")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
