import streamlit as st
import pandas as pd
import numpy as np

# ─── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RFDS QLD B200 Landing Distance Calculator",
    layout="centered"
)
st.title("🛬 RFDS QLD B200 King Air Landing Distance Calculator Unpaved - NOT FOR OPERATIONAL USE")

# ─── Step 1: User Inputs ────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    press_alt = st.slider("Pressure Altitude (ft)", 0, 10000, 0, 250)
    oat       = st.slider("Outside Air Temperature (°C)", -5, 45, 15, 1)
with col2:
    weight = st.slider("Landing Weight (lb)", 9000, 12500, 11500, 100)
    wind   = st.slider(
        "Wind Speed (kt)",
        -10, 30, 0, 1,
        help="Negative = tailwind, Positive = headwind"
    )

# ─── Step 2: Table 1 – Pressure Altitude × OAT (Bilinear Interpolation) ───
raw1 = pd.read_csv("pressureheight_oat.csv", skiprows=[0])
raw1 = raw1.rename(
    columns={raw1.columns[0]: "dummy", raw1.columns[1]: "PressAlt"}
)
tbl1 = raw1.drop(columns=["dummy"]).set_index("PressAlt")
tbl1.columns = tbl1.columns.astype(int)

def lookup_tbl1_bilinear(df, pa, t):
    pas  = np.array(sorted(df.index))
    oats = np.array(sorted(df.columns))
    pa   = np.clip(pa,  pas[0],  pas[-1])
    t    = np.clip(t,   oats[0], oats[-1])
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
st.success(f"Baseline landing distance: **{baseline:.0f} ft**")

# ─── Step 3: Table 2 – Weight Adjustment (1D Interpolation) ────────────────
raw2    = pd.read_csv("weightadjustment.csv", header=0)
wt_cols = [int(w) for w in raw2.columns]
df2     = raw2.astype(float)
df2.columns = wt_cols

def lookup_tbl2_interp(df, baseline, w):
    wt_avail = np.array(sorted(df.columns))
    w_clip   = float(np.clip(w, wt_avail.min(), wt_avail.max()))
    low  = wt_avail[wt_avail <= w_clip].max()
    high = wt_avail[wt_avail >= w_clip].min()
    wt_max    = wt_avail.max()

    # ← Sort rows by the 12500-lb (reference) roll so np.interp sees an ascending X
    tbl       = df.sort_values(by=wt_max, ignore_index=True)
    ref_rolls = tbl[wt_max].values
    d_low     = tbl[low].values  - ref_rolls
    d_high    = tbl[high].values - ref_rolls

    if low == high:
        curve = d_low
    else:
        frac  = (w_clip - low) / (high - low)
        curve = d_low + (d_high - d_low) * frac

    # ← now we can safely interp, with left/right extrapolation if needed
    delta = np.interp(
        baseline,
        ref_rolls,
        curve,
        left=curve[0],
        right=curve[-1]
    )
    return baseline + float(delta)

weight_adj = lookup_tbl2_interp(df2, baseline, weight)
st.markdown("### Step 2: Weight Adjustment")
st.success(f"Weight-adjusted distance: **{weight_adj:.0f} ft**")

# ─── Step 4: Table 3 – Wind Adjustment (1D Interpolation) ─────────────────
raw3      = pd.read_csv("wind adjustment.csv", header=None)
wind_cols = [int(w) for w in raw3.iloc[0]]
df3       = (
    raw3
    .iloc[1:]
    .reset_index(drop=True)
    .apply(pd.to_numeric, errors="coerce")
)
df3.columns = wind_cols

def lookup_tbl3_interp(df, refd, ws):
    ws_avail   = np.array(sorted(df.columns))
    ws_clip    = float(np.clip(ws, ws_avail.min(), ws_avail.max()))
    low  = ws_avail[ws_avail <= ws_clip].max()
    high = ws_avail[ws_avail >= ws_clip].min()
    ref_rolls   = df[0].values
    d_low       = df[low].values  - ref_rolls
    d_high      = df[high].values - ref_rolls

    if low == high:
        curve = d_low
    else:
        frac  = (ws_clip - low) / (high - low)
        curve = d_low + (d_high - d_low) * frac

    delta = np.interp(refd, ref_rolls, curve)
    return float(delta)

delta_wind = lookup_tbl3_interp(df3, weight_adj, wind)
wind_adj   = weight_adj + delta_wind
st.markdown("### Step 3: Wind Adjustment")
st.success(f"After wind adjustment: **{wind_adj:.0f} ft**")

# ─── Step 5: Table 4 – 50 ft Obstacle Correction (1D Interpolation) ────────
raw4 = pd.read_csv("50ft.csv", header=None)
df4  = raw4.iloc[:, :2].copy()
df4.columns = [0, 50]
df4 = (
    df4
    .apply(pd.to_numeric, errors="coerce")
    .dropna()
    .reset_index(drop=True)
)

def lookup_tbl4_interp(df, ground_roll):
    # sort by your ground-roll column
    tbl        = df.sort_values(by=0, ignore_index=True)
    rolls      = tbl[0].values         # ground-roll distances
    obs_totals = tbl[50].values        # this is actually the *increment* to clear 50 ft

    # compute the delta curve (increment)
    deltas     = obs_totals - rolls

    # interpolate that delta at your wind-adjusted roll
    delta50    = np.interp(ground_roll, rolls, deltas)

    # add it on to the ground roll
    return float(ground_roll + delta50)

obs50 = lookup_tbl4_interp(df4, wind_adj)
st.markdown("### Step 4: 50 ft Obstacle Correction")
st.success(f"Final landing distance over 50 ft obstacle: **{obs50:.0f} ft**")

# ─── Additional Output: Distance in Meters ─────────────────────────────────
obs50_m = obs50 * 0.3048
st.markdown("### Final Landing Distance in Meters")
st.success(f"{obs50_m:.1f} m")

# ─── Step 6: Apply a Factor ────────────────────────────────────────────────
factor_options = {
    "Standard Factor Dry (1.43)": 1.43,
    "Standard Factor Wet (1.65)": 1.65,
    "Approved Factor Dry (1.20)": 1.20,
    "Approved Factor Wet (1.38)": 1.38,
}

factor_label = st.selectbox(
    "Select Landing Distance Factor",
    list(factor_options.keys())
)
factor = factor_options[factor_label]

factored_ft = obs50 * factor
factored_m  = factored_ft * 0.3048

st.markdown("### Factored Landing Distance")
c1, c2 = st.columns(2)
c1.success(f"{factored_ft:.0f} ft")
c2.success(f"{factored_m:.1f} m")

# ─── Step X: Runway Slope Input & Adjustment ───────────────────────────────
slope_pct = st.number_input(
    "Runway Slope (%)",
    min_value=-5.0,
    max_value= 0.0,
    value= 0.0,
    step= 0.1,
    help="Negative = downslope (increases distance), Positive = upslope (no effect)"
)
slope_factor = 1.0 + max(-slope_pct, 0.0) * 0.20

sloped_ft = factored_ft * slope_factor
sloped_m  = sloped_ft * 0.3048

st.markdown("### Slope Adjustment")
r1, r2 = st.columns(2)
r1.write(f"**Slope:** {slope_pct:+.1f}%")
r2.write(f"**Slope Factor:** ×{slope_factor:.2f}")

r3, r4 = st.columns(2)
r3.success(f"Distance w/ Slope: **{sloped_ft:.0f} ft**")
r4.success(f"Distance w/ Slope: **{sloped_m:.1f} m**")

# ─── Step Y: Landing Distance Available & Go/No-Go ─────────────────────────
avail_m = st.number_input(
    "Landing Distance Available (m)",
    min_value=0.0,
    value=1150.0,
    step=5.0,
    help="Enter the runway length available in metres"
)
avail_ft = avail_m / 0.3048

st.markdown("### Available Runway Length")
a1, a2 = st.columns(2)
a1.write(f"**{avail_m:.0f} m**")
a2.write(f"**{avail_ft:.0f} ft**")

# Determine if tailwind exists (positive wind value)
has_tailwind = wind < 0

# Check if the 1.20 factor is selected
using_1_2_factor = factor_label == "Approved Factor Dry (1.20)"

# Go/No-Go Decision Logic
st.markdown("### Go/No-Go Decision")

if using_1_2_factor and has_tailwind:
    st.error("❌ Landing not permitted: No tailwind component permitted with 1.2")
elif avail_ft >= sloped_ft:
    st.success("✅ Enough runway available for landing")
else:
    st.error("❌ Insufficient runway available for landing")

st.markdown("### Data extracted from B200-601-228 HFG Performance Landing Distance w Propeller Reversing - Flap 100%")
st.markdown("Created by H Watson and R Thomas")
