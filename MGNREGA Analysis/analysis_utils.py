import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    # Build panel id
    df["region_id"] = (
        df["District"].astype(str).str.strip()
        + "_"
        + df["State"].astype(str).str.strip()
    )
    df["year"] = df["Year"].astype(int)

    # Keep only balanced regions (no missing years)
    years = sorted(df["year"].dropna().unique().tolist())
    n_years = len(years)
    per_region_years = df.groupby("region_id")["year"].nunique()
    valid_regions = per_region_years[per_region_years == n_years].index
    df = df[df["region_id"].isin(valid_regions)].copy()

    # Standardized field names used in models
    df["rainfall"] = df["Annual_Rainfall_mm"]
    df["income"] = df["Total_Income"]
    df["agri_yield"] = df["agri_yield_index"]
    df["population"] = df["Rural_Population"]

    # MGNREGA intensity proxy
    denom = df["Registered_HH"].replace(0, np.nan)
    df["mgnrega"] = df["Total_Persondays"] / denom

    # Shares for distributional effects
    worker_total = df["Workers_Total"].replace(0, np.nan)
    df["scst_share"] = (df["Workers_SC"] + df["Workers_ST"]) / worker_total

    emp_availed = df["Employment_Availed"].replace(0, np.nan)
    df["women_share"] = df["Women_Employment_Provided"] / emp_availed

    # Region-specific rainfall shock
    region_mean = df.groupby("region_id")["rainfall"].transform("mean")
    region_std = df.groupby("region_id")["rainfall"].transform("std").replace(0, np.nan)
    df["rainfall_shock"] = (df["rainfall"] - region_mean) / region_std

    # Lag needed for structural transformation
    df = df.sort_values(["region_id", "year"])
    df["mgnrega_lag1"] = df.groupby("region_id")["mgnrega"].shift(1)

    # Winsorize numeric columns at 1% tails
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            df[col] = pd.Series(
                winsorize(df[col].astype(float), limits=[0.01, 0.01]),
                index=df.index,
            )

    return df


def fit_fe_clustered(formula: str, df: pd.DataFrame):
    work = df.copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    model = smf.ols(formula=formula, data=work)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": work["region_id"]})
    return results, work


def minmax_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    smin, smax = s.min(), s.max()
    if pd.isna(smin) or pd.isna(smax) or smin == smax:
        return pd.Series(0.5, index=s.index)
    return (s - smin) / (smax - smin)


def compute_vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    x = df[cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    x = x.assign(constant=1.0)
    out = []
    for i, col in enumerate(x.columns):
        if col == "constant":
            continue
        out.append((col, variance_inflation_factor(x.values, i)))
    return pd.DataFrame(out, columns=["variable", "vif"]).sort_values("vif", ascending=False)


def summarize_model(results, key_terms: list) -> pd.DataFrame:
    rows = []
    for term in key_terms:
        if term in results.params.index:
            rows.append(
                {
                    "term": term,
                    "coef": results.params[term],
                    "std_err": results.bse[term],
                    "p_value": results.pvalues[term],
                }
            )
    return pd.DataFrame(rows)
