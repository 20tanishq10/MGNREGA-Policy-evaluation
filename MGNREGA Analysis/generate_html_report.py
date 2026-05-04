from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
FINAL_DIR = ROOT / "final_outputs_refined"
REPORT_PATH = ROOT / "MGNREGA_Impact_Report.html"
PILLARS = ["shock", "distribution", "income", "distortion", "structural_persistence"]


def fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return x


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "<p><em>Not available</em></p>"
    show = df.head(max_rows).copy()
    for c in show.columns:
        show[c] = show[c].map(fmt)
    return show.to_html(index=False, classes="table", border=0)


def radar_plot(values_dict: dict, title: str, outpath: Path):
    labels = list(values_dict.keys())
    vals = [min(1.0, max(0.0, float(values_dict[k]))) for k in labels]
    vals += vals[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.title() for k in labels])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def build_time_trend(composite: pd.DataFrame) -> pd.DataFrame:
    if composite.empty:
        return pd.DataFrame()
    trend = (
        composite.groupby("year", as_index=False)[PILLARS + ["index_equal_rescaled", "index_weighted_rescaled"]]
        .mean()
        .sort_values("year")
    )
    return trend


def build_covid(composite: pd.DataFrame) -> pd.DataFrame:
    if composite.empty:
        return pd.DataFrame()
    x = composite.copy()
    x["period"] = np.where(x["year"].between(2015, 2019), "Pre-COVID", "Post-COVID")
    return x.groupby("period", as_index=False)[PILLARS + ["index_equal_rescaled", "index_weighted_rescaled"]].mean()


def generate_revised_figures(time_trend: pd.DataFrame, covid: pd.DataFrame, composite: pd.DataFrame):
    if composite.empty or time_trend.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(time_trend["year"], time_trend["index_equal_rescaled"], marker="o", label="Equal Weight")
    plt.plot(time_trend["year"], time_trend["index_weighted_rescaled"], marker="o", label="Weighted")
    plt.title("Time Trend: Composite Index")
    plt.xlabel("Year")
    plt.ylabel("Index")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "refined_time_trend_index.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for p in PILLARS:
        plt.plot(time_trend["year"], time_trend[p], marker="o", label=p.title())
    plt.title("Time Trend: Pillar Scores")
    plt.xlabel("Year")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "refined_time_trend_pillars.png", dpi=300)
    plt.close()

    if not covid.empty and "period" in covid.columns:
        covid.set_index("period")[PILLARS].T.plot(kind="bar", figsize=(10, 6))
        plt.title("Pre vs Post COVID: Pillar Scores")
        plt.xlabel("Pillar")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FINAL_DIR / "refined_pre_post_covid_pillars.png", dpi=300)
        plt.close()

    overall = composite[PILLARS].mean().to_dict()
    pre = composite.loc[composite["year"].between(2015, 2019), PILLARS].mean().to_dict()
    post = composite.loc[composite["year"].between(2020, 2024), PILLARS].mean().to_dict()
    radar_plot(overall, "Radar: Overall", FINAL_DIR / "refined_radar_overall.png")
    radar_plot(pre, "Radar: Pre-COVID", FINAL_DIR / "refined_radar_pre_covid.png")
    radar_plot(post, "Radar: Post-COVID", FINAL_DIR / "refined_radar_post_covid.png")


def main():
    composite = read_csv(FINAL_DIR / "composite_index_rescaled.csv")
    heterogeneity = read_csv(FINAL_DIR / "heterogeneity_advanced.csv")
    dynamic = read_csv(FINAL_DIR / "dynamic_effects.csv")
    checks = read_csv(FINAL_DIR / "validation_checks.csv")

    time_trend = build_time_trend(composite)
    covid = build_covid(composite)
    corr = composite[PILLARS + ["index_equal_rescaled"]].corr() if not composite.empty else pd.DataFrame()
    generate_revised_figures(time_trend, covid, composite)

    n_regions = composite["region_id"].nunique() if not composite.empty else 0
    years = sorted(composite["year"].dropna().astype(int).unique().tolist()) if "year" in composite.columns else []
    yr_text = f"{years[0]}-{years[-1]}" if years else "N/A"
    max_offdiag_corr = np.nan
    if not corr.empty:
        corr_num = corr.select_dtypes(include=[np.number])
        vals = corr_num.values.astype(float)
        max_offdiag_corr = np.max(np.abs(vals - np.eye(vals.shape[0])))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MGNREGA Impact Report (Publishable)</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; line-height: 1.45; }}
    h1, h2, h3 {{ color: #0f2742; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; background: #fbfdff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .kpi {{ font-size: 28px; font-weight: 700; color: #113d6b; }}
    .muted {{ color: #546e7a; }}
    .table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 16px; }}
    .table th, .table td {{ border: 1px solid #dfe7ef; padding: 6px 8px; text-align: left; }}
    .table th {{ background: #eef4fb; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; border-radius: 6px; }}
    code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>MGNREGA Impact Study Report (Publishable Refined Version)</h1>
  <p class="muted">Built from final refinement outputs with improved shock identification, rescaled index design, dynamic response, and validation checks.</p>
  <div class="grid">
    <div class="card"><div class="muted">Regions</div><div class="kpi">{n_regions}</div></div>
    <div class="card"><div class="muted">Period</div><div class="kpi">{yr_text}</div></div>
    <div class="card"><div class="muted">Max |Pillar Corr|</div><div class="kpi">{fmt(max_offdiag_corr)}</div></div>
  </div>
  <div class="card">
    <h3>Method Summary (Final Refinement)</h3>
    <ul>
      <li>Refined continuous + extreme shock specification for Pillar 1.</li>
      <li>Pillar-wise z-standardization then min-max rescaling before index construction.</li>
      <li>Distortion uses elasticity ratio and welfare inversion.</li>
      <li>Structural pillar interpreted as persistence via lagged MGNREGA-income relation.</li>
      <li>Advanced heterogeneity and dynamic lag/current/lead response included.</li>
    </ul>
  </div>
  <h2>Validation Checks</h2>
  {html_table(checks, 20)}
  <h2>Time Evolution</h2>
  {html_table(time_trend, 20)}
  <div class="grid">
    <div><img src="final_outputs_refined/refined_time_trend_index.png" alt="Refined Time Trend Index" /></div>
    <div><img src="final_outputs_refined/refined_time_trend_pillars.png" alt="Refined Time Trend Pillars" /></div>
  </div>
  <h2>COVID Comparison</h2>
  {html_table(covid, 10)}
  <img src="final_outputs_refined/refined_pre_post_covid_pillars.png" alt="Refined COVID Comparison" />
  <h2>Radar Profiles</h2>
  <div class="grid">
    <div><img src="final_outputs_refined/refined_radar_overall.png" alt="Refined Radar Overall" /></div>
    <div><img src="final_outputs_refined/refined_radar_pre_covid.png" alt="Refined Radar Pre" /></div>
    <div><img src="final_outputs_refined/refined_radar_post_covid.png" alt="Refined Radar Post" /></div>
  </div>
  <h2>Heterogeneity Analysis</h2>
  {html_table(heterogeneity, 20)}
  <h2>Dynamic Response</h2>
  {html_table(dynamic, 10)}
  <h2>Correlation Matrix</h2>
  {html_table(corr, 20)}
  <h2>Core Output Files</h2>
  <ul>
    <li><code>final_outputs_refined/pillar1_shock_score_refined.csv</code></li>
    <li><code>final_outputs_refined/pillar4_distortion_score_refined.csv</code></li>
    <li><code>final_outputs_refined/pillar5_structural_persistence_score.csv</code></li>
    <li><code>final_outputs_refined/composite_index_rescaled.csv</code></li>
    <li><code>final_outputs_refined/heterogeneity_advanced.csv</code></li>
    <li><code>final_outputs_refined/dynamic_effects.csv</code></li>
    <li><code>final_outputs_refined/validation_checks.csv</code></li>
  </ul>
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Report generated: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
