import streamlit as st
import pandas as pd
import numpy as np

# ─── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Unpaved B200 Landing Distance Calculator",
    layout="centered"
)
st.title("🛬 Unpaved B200 Landing Distance Estimator")

# ─── Step 1: User Inputs ────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    press_alt = st.slider("Pressure Altitude (ft)",   0, 10000, 2000, 250)
    oat       = st.slider("Outside Air Temperature (°C)", -5, 45, 15, 1)
with col2:
    weight = st.slider("Landing Weight (lb)", 9000, 12500, 11500, 100)
    wind   = st.slider("Wind Speed (kt)",     -20,    30,    0,   1,
                       help="Negative = tailwind, Positive = headwind")

# ─── Step 2: Baseline (Pressure Alt × OAT) ─────────────────────────────────
raw1 = pd.read_csv("pressureheight_oat.csv", skiprows=[0])
raw1 = raw1.rename(
    columns={raw1.columns[0]: "dummy", raw1.columns[1]: "PressAlt"}
)
tbl1 = raw1.drop(columns=["dummy"]).set_index("PressAlt")
tbl1.columns = tbl1.columns.astype(int)

def lookup_tbl1_bilinear(df, pa, t):
    pas  = np.array(sorted(df.index))
    oats = np.array(sorted(df.columns))
    pa   = np.clip(pa, pas[0], pas[-1])
    t    = np.clip(t,  oats[0], oats[-1])
    x1 = pas[pas <= pa].max(); x2 = pas[pas >= pa].min()
    y1 = oats[oats <= t].max(); y2 = oats[oats >= t].min()
    Q11 = df.at[x1, y1]; Q21 = df.at[x2, y1]
    Q12 = df.at[x1, y2]; Q22 = df.at[x2, y2]
    if x1 == x2 and y1 == y2:
        return Q11
    if x1 == x2:
        return Q11 + (Q12 - Q11) * (t - y1) / (y2 - y1)
    if y1 == y2:
        return Q11 + (Q21 - Q11) * (pa - x1) / (x2 - x1)
    denom = (x2 - x1) * (y2 - y1)
    fxy1  = Q11 * (x2 - pa) + Q21 * (pa - x1)
    fxy2  = Q12 * (x2 - pa) + Q22 * (pa - x1)
    return (fxy1 * (y2 - t) + fxy2 * (t - y1)) / denom

baseline = lookup_tbl1_bilinear(tbl1, press_alt, oat)
st.markdown("### Step 1: Baseline Distance")
st.success(f"{baseline:.0f} ft")

# ─── Step 3: Weight Adjustment (1D Interpolation) ─────────────────────────
# Load small table, first row = weights, remaining rows = adjustment values
raw2 = pd.read_csv("weightadjustment.csv", header=None)
weight_cols = raw2.iloc[0].astype(int).tolist()
df2 = raw2.iloc[1:].reset_index(drop=True).apply(pd.to_numeric)
df2.columns = weight_cols

def lookup_weight_interp(df, baseline, w):
    # sorted available weights
    w_avail = np.array(sorted(df.columns))
    # clamp slider weight
    w = float(np.clip(w, w_avail.min(), w_avail.max()))
    # bracket
    low  = w_avail[w_avail <= w].max()
    high = w_avail[w_avail >= w].min()
    # reference roll at heaviest weight
    ref_col = w_avail.max()
    ref_rolls = df[ref_col].values
    # compute delta curves
    d_low  = df[low].values - ref_rolls
    d_high = df[high].values - ref_rolls
    # interpolate between the two curves
    if low == high:
        curve = d_low
    else:
        frac = (w - low) / (high - low)
        curve = d_low + (d_high - d_low) * frac
    # get exact delta at our baseline
    delta = np.interp(baseline, ref_rolls, curve)
    return baseline + float(delta)

weight_adj = lookup_weight_interp(df2, baseline, weight)
st.markdown("### Step 2: Weight Adjustment")
st.success(f"{weight_adj:.0f} ft")

# ─── Step 4: Wind Adjustment (1D Interpolation) ───────────────────────────
raw3 = pd.read_csv("wind adjustment.csv", header=None)
wind_cols = raw3.iloc[0].astype(int).tolist()
df3 = raw3.iloc[1:].reset_index(drop=True).apply(pd.to_numeric)
df3.columns = wind_cols

def lookup_wind_interp(df, refd, ws):
    w_avail = np.array(sorted(df.columns))
    ws = float(np.clip(ws, w_avail.min(), w_avail.max()))
    low  = w_avail[w_avail <= ws].max()
    high = w_avail[w_avail >= ws].min()
    # reference roll at zero wind is in column '0'
    ref_rolls = df[0].values
    d_low  = df[low].values - ref_rolls
    d_high = df[high].values - ref_rolls
    if low == high:
        curve = d_low
    else:
        frac = (ws - low) / (high - low)
        curve = d_low + (d_high - d_low) * frac
    delta = np.interp(refd, ref_rolls, curve)
    return float(delta)

delta_wind = lookup_wind_interp(df3, weight_adj, wind)
wind_adj = weight_adj + delta_wind
st.markdown("### Step 3: Wind Adjustment")
st.success(f"{wind_adj:.0f} ft")

# ─── Step 5: 50 ft Obstacle (1D Interpolation) ─────────────────────────────
raw4 = pd.read_csv("50ft.csv", header=None)
df4  = raw4.iloc[:, :2].copy()
df4.columns = [0, 50]
df4 = df4.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

def lookup_obstacle(df, refd):
    ref_rolls = df[0].values
    obs_vals  = df[50].values
    return float(np.interp(refd, ref_rolls, obs_vals))

obs50 = lookup_obstacle(df4, wind_adj)
st.markdown("### Step 4: 50 ft Obstacle Correction")
st.success(f"{obs50:.0f} ft")

# ─── Final: Show in Metres ─────────────────────────────────────────────────
obs50_m = obs50 * 0.3048
st.markdown("### Over-50 ft Distance")
colA, colB = st.columns(2)
colA.metric("Feet", f"{obs50:.0f} ft")
colB.metric("Metres", f"{obs50_m:.1f} m")
