from pathlib import Path
import pandas as pd

from analysis_utils import (
    load_and_preprocess,
    fit_fe_clustered,
    minmax_normalize,
    compute_vif,
    summarize_model,
)


DATA_PATH = Path("Panel_Data 2014-24.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def save_pillar1(df: pd.DataFrame):
    df = df.copy()
    df["mgnrega_x_shock"] = df["mgnrega"] * df["rainfall_shock"]
    formula = "income ~ mgnrega + rainfall_shock + mgnrega:rainfall_shock + C(region_id) + C(year)"
    res, used = fit_fe_clustered(formula, df)

    summarize_model(res, ["mgnrega", "rainfall_shock", "mgnrega:rainfall_shock"]).to_csv(
        OUT_DIR / "pillar1_coefficients.csv", index=False
    )
    compute_vif(used, ["mgnrega", "rainfall_shock", "mgnrega_x_shock"]).to_csv(
        OUT_DIR / "pillar1_vif.csv", index=False
    )

    beta3 = res.params.get("mgnrega:rainfall_shock", 0.0)
    raw = used["mgnrega_x_shock"] * beta3
    score = minmax_normalize(raw)
    out = used[["region_id", "year"]].copy()
    out["pillar1_shock_score"] = score.values
    out = out.sort_values(["region_id", "year"])
    out.to_csv(OUT_DIR / "pillar1_shock_score.csv", index=False)


def save_pillar2(df: pd.DataFrame):
    df = df.copy()
    df["mgnrega_x_scst"] = df["mgnrega"] * df["scst_share"]
    df["mgnrega_x_women"] = df["mgnrega"] * df["women_share"]
    formula = "income ~ mgnrega + mgnrega:scst_share + mgnrega:women_share + rainfall + C(region_id) + C(year)"
    res, used = fit_fe_clustered(formula, df)

    summarize_model(res, ["mgnrega", "mgnrega:scst_share", "mgnrega:women_share", "rainfall"]).to_csv(
        OUT_DIR / "pillar2_coefficients.csv", index=False
    )
    compute_vif(
        used, ["mgnrega", "scst_share", "women_share", "mgnrega_x_scst", "mgnrega_x_women", "rainfall"]
    ).to_csv(OUT_DIR / "pillar2_vif.csv", index=False)

    b2 = res.params.get("mgnrega:scst_share", 0.0)
    b3 = res.params.get("mgnrega:women_share", 0.0)
    raw = (b2 * used["mgnrega_x_scst"] + b3 * used["mgnrega_x_women"]) / 2.0
    score = minmax_normalize(raw)
    out = used[["region_id", "year"]].copy()
    out["pillar2_distribution_score"] = score.values
    out = out.sort_values(["region_id", "year"])
    out.to_csv(OUT_DIR / "pillar2_distribution_score.csv", index=False)


def save_pillar3(df: pd.DataFrame):
    formula = "income ~ mgnrega + rainfall + population + C(region_id) + C(year)"
    res, used = fit_fe_clustered(formula, df)

    summarize_model(res, ["mgnrega", "rainfall", "population"]).to_csv(
        OUT_DIR / "pillar3_coefficients.csv", index=False
    )
    compute_vif(used, ["mgnrega", "rainfall", "population"]).to_csv(
        OUT_DIR / "pillar3_vif.csv", index=False
    )

    b1 = res.params.get("mgnrega", 0.0)
    raw = b1 * used["mgnrega"]
    score = minmax_normalize(raw)
    out = used[["region_id", "year"]].copy()
    out["pillar3_income_score"] = score.values
    out = out.sort_values(["region_id", "year"])
    out.to_csv(OUT_DIR / "pillar3_income_score.csv", index=False)


def save_pillar4(df: pd.DataFrame):
    res_income, used_income = fit_fe_clustered(
        "income ~ mgnrega + rainfall + C(region_id) + C(year)", df
    )
    res_yield, used_yield = fit_fe_clustered(
        "agri_yield ~ mgnrega + rainfall + C(region_id) + C(year)", df
    )

    b1 = res_income.params.get("mgnrega", 0.0)
    b2 = res_yield.params.get("mgnrega", 0.0)

    coef_table = pd.DataFrame(
        [
            {"model": "income", "term": "mgnrega", "coef": b1, "p_value": res_income.pvalues.get("mgnrega")},
            {"model": "yield", "term": "mgnrega", "coef": b2, "p_value": res_yield.pvalues.get("mgnrega")},
            {"model": "combined", "term": "distortion", "coef": b1 - b2, "p_value": None},
        ]
    )
    coef_table.to_csv(OUT_DIR / "pillar4_coefficients.csv", index=False)
    compute_vif(used_income, ["mgnrega", "rainfall"]).to_csv(OUT_DIR / "pillar4_vif_income.csv", index=False)
    compute_vif(used_yield, ["mgnrega", "rainfall"]).to_csv(OUT_DIR / "pillar4_vif_yield.csv", index=False)

    merged = used_income[["region_id", "year", "mgnrega"]].merge(
        used_yield[["region_id", "year"]], on=["region_id", "year"], how="inner"
    )
    distortion_raw = (b1 - b2) * merged["mgnrega"]
    distortion_norm = minmax_normalize(distortion_raw)
    score = 1 - distortion_norm  # Higher score means lower distortion
    out = merged[["region_id", "year"]].copy()
    out["pillar4_distortion_score"] = score.values
    out = out.sort_values(["region_id", "year"])
    out.to_csv(OUT_DIR / "pillar4_distortion_score.csv", index=False)


def save_pillar5(df: pd.DataFrame):
    res_yield, used_yield = fit_fe_clustered(
        "agri_yield ~ mgnrega_lag1 + rainfall + C(region_id) + C(year)", df
    )
    res_income, used_income = fit_fe_clustered(
        "income ~ mgnrega_lag1 + rainfall + C(region_id) + C(year)", df
    )

    coef_table = pd.DataFrame(
        [
            {
                "model": "yield",
                "term": "mgnrega_lag1",
                "coef": res_yield.params.get("mgnrega_lag1", 0.0),
                "p_value": res_yield.pvalues.get("mgnrega_lag1"),
            },
            {
                "model": "income",
                "term": "mgnrega_lag1",
                "coef": res_income.params.get("mgnrega_lag1", 0.0),
                "p_value": res_income.pvalues.get("mgnrega_lag1"),
            },
        ]
    )
    coef_table.to_csv(OUT_DIR / "pillar5_coefficients.csv", index=False)
    compute_vif(used_yield, ["mgnrega_lag1", "rainfall"]).to_csv(OUT_DIR / "pillar5_vif_yield.csv", index=False)
    compute_vif(used_income, ["mgnrega_lag1", "rainfall"]).to_csv(OUT_DIR / "pillar5_vif_income.csv", index=False)

    beta = res_yield.params.get("mgnrega_lag1", 0.0)
    raw = beta * used_yield["mgnrega_lag1"]
    score = minmax_normalize(raw)
    out = used_yield[["region_id", "year"]].copy()
    out["pillar5_structural_score"] = score.values
    out = out.sort_values(["region_id", "year"])
    out.to_csv(OUT_DIR / "pillar5_structural_score.csv", index=False)


def main():
    df = load_and_preprocess(str(DATA_PATH))
    save_pillar1(df)
    save_pillar2(df)
    save_pillar3(df)
    save_pillar4(df)
    save_pillar5(df)
    print("Corrected Step 2 outputs written with region-year granularity.")


if __name__ == "__main__":
    main()
