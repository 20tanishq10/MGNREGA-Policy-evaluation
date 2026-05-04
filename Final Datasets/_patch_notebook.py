"""
Patch panel_data_construction.ipynb:
1. Insert a new Cell 0 that interpolates SC-ST data for Karnataka, West Bengal, Ladakh
2. Re-clean cells 15 and 17
"""

import json, os

NB_PATH = os.path.join(os.path.dirname(__file__), "panel_data_construction.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── New Cell 0: SC-ST pre-interpolation ─────────────────────────────────────
interpolation_source = """\
# ═══════════════════════════════════════════════════════════════════════════
# PRE-STEP: Interpolate missing SC-ST values for Karnataka, West Bengal, Ladakh
# ═══════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import re, os

# FIX: Use working directory (not __file__)
DATA_DIR = os.getcwd()

def norm(s):
    if pd.isna(s): return np.nan
    return re.sub(r'\\s+', ' ', str(s).strip().upper())

sc_st = pd.read_csv(os.path.join(DATA_DIR, 'mgnrega_sc-st.csv'))
sc_st.columns = sc_st.columns.str.strip()
sc_st['District'] = sc_st['District'].apply(norm)
sc_st['State']    = sc_st['State'].apply(norm)

sc_st['Year'] = sc_st['Year'].apply(
    lambda x: int(re.match(r'(\\d{4})', str(x)).group(1))
    if pd.notnull(x) and re.match(r'(\\d{4})', str(x)) else np.nan
).astype('Int64')

TARGET_YEARS = list(range(2014, 2025))
VALUE_COLS   = ['Women_Employment_Provided', 'Workers_SC', 'Workers_ST', 'Workers_Total']

# ── West Bengal: district consolidation ─────────────────────────────────────
WB_ALIAS = {
    '24 PARGANAS (NORTH)' : 'NORTH TWENTY FOUR PARGANAS',
    '24 PARGANAS SOUTH'   : 'SOUTH TWENTY FOUR PARGANAS',
    'NORTH 24 PARGANAS'   : 'NORTH TWENTY FOUR PARGANAS',
    'SOUTH 24 PARGANAS'   : 'SOUTH TWENTY FOUR PARGANAS',
    'DINAJPUR DAKSHIN'    : 'DAKSHIN DINAJPUR',
    'DINAJPUR UTTAR'      : 'UTTAR DINAJPUR',
    'DARJEELING GORKHA HILL COUNCIL (DGHC)'  : 'DARJEELING',
    'GORKHALAND TERRITORIAL ADMINISTRATION (GTA)': 'DARJEELING',
    'PURBA BARDHAMAN'     : 'BARDHAMAN',
    'PASCHIM BARDHAMAN'   : 'BARDHAMAN',
}

wb_mask = sc_st['State'] == 'WEST BENGAL'
sc_st.loc[wb_mask, 'District'] = sc_st.loc[wb_mask, 'District'].map(
    lambda x: WB_ALIAS.get(x, x)
)

# Aggregate duplicates after aliasing
sc_st = sc_st.groupby(['District', 'State', 'Year'], as_index=False)[VALUE_COLS].sum(min_count=1)

# ── Helper: interpolate a state's missing district-years ─────────────────────
def interpolate_state(df, state):
    sub    = df[df['State'] == state].copy()
    others = df[df['State'] != state].copy()

    # SAFETY CHECK (important for Ladakh)
    if sub['Year'].nunique() < 3:
        print(f"WARNING: Too few data points for reliable interpolation: {state}")

    districts = sub['District'].unique()

    full_idx = pd.MultiIndex.from_product(
        [districts, TARGET_YEARS], names=['District', 'Year']
    )
    full = pd.DataFrame(index=full_idx).reset_index()
    full['State'] = state

    merged = full.merge(sub, on=['District', 'State', 'Year'], how='left')

    for col in VALUE_COLS:
        merged[col] = merged.groupby('District')[col].transform(
            lambda s: s.interpolate(method='linear', limit_direction='both')
        )
        merged[col] = merged[col].clip(lower=0).round(0)

    return pd.concat([others, merged], ignore_index=True)

# Apply interpolation
for state in ['KARNATAKA', 'WEST BENGAL', 'LADAKH']:
    before = sc_st[sc_st['State'] == state].shape[0]
    sc_st = interpolate_state(sc_st, state)
    after  = sc_st[sc_st['State'] == state].shape[0]
    print(f'{state}: rows {before} -> {after}')

# ── Verification ────────────────────────────────────────────────────────────
print('\\nPost-interpolation check:')
for state in ['KARNATAKA', 'WEST BENGAL', 'LADAKH']:
    sub = sc_st[sc_st['State'] == state]
    for dist in sub['District'].unique():
        years_present = sorted(sub[sub['District'] == dist]['Year'].tolist())
        missing = [y for y in TARGET_YEARS if y not in years_present]
        if missing:
            print(f'MISSING: {state} / {dist}: {missing}')

# ── Save (OVERWRITE as requested) ───────────────────────────────────────────
out_path = os.path.join(DATA_DIR, 'mgnrega_sc-st.csv')
sc_st.to_csv(out_path, index=False)

print(f'\\nSaved (overwritten): {out_path}')
print(f'Total rows: {len(sc_st)}')
"""

new_cell_0 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "a0000000",
    "metadata": {},
    "outputs": [],
    "source": interpolation_source
}

# Insert at top
nb["cells"].insert(0, new_cell_0)

# ── Fix Cell 15 (MPI + population) ──────────────────────────────────────────
for cell in nb["cells"]:
    if cell.get("id") == "a0000015":
        cell["source"] = cell["source"].replace(
            "pop_filtered = pop_filtered.drop_duplicates(subset=[d], keep='first')",
            "pop_filtered = pop_filtered.groupby(d).mean().reset_index()"
        )
        cell["execution_count"] = None
        cell["outputs"] = []

# ── Fix Cell 17 (soft dedup already correct) ────────────────────────────────
for cell in nb["cells"]:
    if cell.get("id") == "a0000017":
        cell["execution_count"] = None
        cell["outputs"] = []

# Save notebook
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")