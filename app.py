"""
Audit Risk Intelligence — Production-grade Streamlit app
=========================================================
Features:
  • Session-based transaction history log with sparkline trend
  • Batch CSV upload for bulk scoring
  • SHAP waterfall explainability per prediction
  • st.secrets-based password authentication
  • Structured prediction logging (predictions.jsonl)
  • Polished dark UI with animated score reveal
"""

import json
import os
import time
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Optional — SHAP is gracefully degraded if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
LOW_RISK_THRESHOLD  = 0.30
HIGH_RISK_THRESHOLD = 0.70
HIGH_AMOUNT         = 50_000
HIGH_VENDOR_RISK    = 0.70
LOG_FILE            = "predictions.jsonl"
BATCH_REQUIRED_COLS = [
    "Transaction_Amount", "Vendor_Risk_Score",
    "Previous_Fraud_Flag", "Unusual_Time_Flag",
]

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Audit Risk Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# AUTH  (password stored in .streamlit/secrets.toml)
# ─────────────────────────────────────────────
def check_auth() -> bool:
    """Return True if auth is disabled or user has authenticated."""
    try:
        required = st.secrets.get("app_password")
    except Exception:
        required = None

    if not required:
        return True  # auth not configured → open access

    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <div style='max-width:380px;margin:6rem auto;text-align:center;'>
        <p style='font-size:2rem;margin-bottom:0.5rem;'>🔐</p>
        <h2 style='font-family:serif;margin-bottom:1.5rem;'>Audit Risk Intelligence</h2>
    </div>
    """, unsafe_allow_html=True)

    pwd = st.text_input("Enter access password", type="password")
    if st.button("Sign In"):
        if pwd == required:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_auth():
    st.stop()

# ─────────────────────────────────────────────
# MODEL LOADING  (cached — runs once only)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        m   = joblib.load("xgb_audit_model.pkl")
        col = joblib.load("model_columns.pkl")
        return m, col
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        st.stop()

model, COLUMNS = load_model()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {time, amount, prob, tier}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_input_df(
    amount: float, vendor_risk: float,
    prev_fraud: int, unusual: int,
    freq: int, round_amt: int, weekend: int,
) -> pd.DataFrame:
    row = {
        "Transaction_Amount":   amount,
        "Vendor_Risk_Score":    vendor_risk,
        "Transaction_Frequency": freq,
        "Round_Amount_Flag":    round_amt,
        "Weekend_Transaction":  weekend,
        "Previous_Fraud_Flag":  prev_fraud,
        "Unusual_Time_Flag":    unusual,
        "Multiple_Approvals":   0,
        "Vendor_New_Flag":      0,
        # Default one-hot profile (most common in training data)
        "Transaction_Type_Transfer": 1,
        "Department_Finance":        1,
        "Approval_Level_Manager":    1,
    }
    df = pd.DataFrame([row])
    df = df.reindex(columns=COLUMNS, fill_value=0)
    return df.astype(float)


def predict_risk(df: pd.DataFrame) -> float:
    return float(model.predict_proba(df)[0][1])


def risk_tier(prob: float) -> tuple[str, str, str]:
    if prob < LOW_RISK_THRESHOLD:
        return "Low Risk",    "low",    "#34d399"
    elif prob < HIGH_RISK_THRESHOLD:
        return "Medium Risk", "medium", "#fbbf24"
    else:
        return "High Risk",   "high",   "#f87171"


def gather_signals(
    amount, vendor_risk, prev_fraud, unusual, round_amt, weekend
) -> list[str]:
    s = []
    if amount     > HIGH_AMOUNT:       s.append(f"High amount (₹{amount:,.0f})")
    if vendor_risk > HIGH_VENDOR_RISK:  s.append(f"Elevated vendor risk ({vendor_risk:.2f})")
    if prev_fraud:                      s.append("Prior fraud history")
    if unusual:                         s.append("Unusual timing")
    if round_amt:                       s.append("Round-number amount")
    if weekend:                         s.append("Weekend transaction")
    return s


def log_prediction(record: dict):
    """Append a prediction record to the JSONL audit log."""
    record["logged_at"] = datetime.datetime.utcnow().isoformat()
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging failure must never crash the app


def compute_shap(df: pd.DataFrame):
    """Return SHAP values array and feature names, or None if unavailable."""
    if not SHAP_AVAILABLE:
        return None, None
    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(df)
        # XGBoost binary: vals may be 2D list
        if isinstance(vals, list):
            vals = vals[1]
        return vals[0], list(df.columns)
    except Exception:
        return None, None


def batch_score(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Score every row of a raw uploaded DataFrame."""
    results = []
    for _, row in df_raw.iterrows():
        idf = build_input_df(
            amount     = float(row.get("Transaction_Amount", 0)),
            vendor_risk= float(row.get("Vendor_Risk_Score", 0)),
            prev_fraud = int(row.get("Previous_Fraud_Flag", 0)),
            unusual    = int(row.get("Unusual_Time_Flag", 0)),
            freq       = int(row.get("Transaction_Frequency", 5)),
            round_amt  = int(row.get("Round_Amount_Flag", 0)),
            weekend    = int(row.get("Weekend_Transaction", 0)),
        )
        prob = predict_risk(idf)
        tier, _, _ = risk_tier(prob)
        results.append({**row.to_dict(), "Risk_Score_%": round(prob * 100, 2), "Risk_Tier": tier})
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main { background: #07090f; }
.block-container { padding-top: 3.5rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { background: #0c0f1a; border-right: 1px solid #1a2240; }

/* Header */
.ari-logo   { font-family:'Syne',sans-serif; font-weight:800; font-size:1.35rem; color:#e2e8ff; letter-spacing:-0.02em; }
.ari-tagline{ font-size:11px; color:#3d5099; font-weight:500; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px; }

/* Page title */
.page-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem; color:#e2e8ff; margin:0 0 0.2rem; line-height:1.15; }
.page-sub   { font-size:13px; color:#3d5099; }

/* Cards */
.card {
    background:#0c0f1a;
    border:1px solid #1a2240;
    border-radius:14px;
    padding:1.25rem 1.5rem;
    margin-bottom:1rem;
}
.card-label {
    font-size:10px; font-weight:600; letter-spacing:0.16em;
    text-transform:uppercase; color:#3d5099; margin-bottom:0.85rem;
}

/* Score */
.score-big {
    font-family:'Syne',sans-serif !important; font-weight:800 !important;
    font-size:3.8rem; line-height:1; letter-spacing:-0.03em;
}
.score-pct { font-size:1.6rem; font-family:'Syne',sans-serif !important; }
.risk-bar-bg  { height:6px; background:#131929; border-radius:99px; margin:0.9rem 0 0.5rem; overflow:hidden; }
.risk-bar-fill{ height:100%; border-radius:99px; }

/* Badge */
.badge {
    display:inline-flex; align-items:center; gap:6px;
    padding:4px 14px; border-radius:999px;
    font-size:12px; font-weight:600; letter-spacing:0.04em;
}
.badge-low    { background:#0a2318; color:#34d399; border:1px solid #145c38; }
.badge-medium { background:#211800; color:#fbbf24; border:1px solid #5c4010; }
.badge-high   { background:#210a0a; color:#f87171; border:1px solid #5c1818; }

/* Signals */
.sig-pill {
    display:inline-flex; align-items:center; gap:5px;
    background:#110d1e; border:1px solid #2a1f45;
    color:#a78bfa; font-size:11px; font-weight:500;
    padding:3px 10px; border-radius:999px;
}

/* Rec banner */
.rec { display:flex; align-items:center; gap:10px; padding:12px 16px; border-radius:10px; font-size:13px; font-weight:500; }
.rec-audit{ background:#210a0a; border:1px solid #5c1818; color:#f87171; }
.rec-clear{ background:#0a2318; border:1px solid #145c38; color:#34d399; }

/* History table */
.htable { width:100%; border-collapse:collapse; font-size:12px; }
.htable th { color:#3d5099; font-weight:600; padding:4px 8px; border-bottom:1px solid #1a2240; text-align:left; font-size:10px; letter-spacing:0.1em; text-transform:uppercase; }
.htable td { padding:6px 8px; border-bottom:1px solid #0f1525; color:#9ba8cc; }
.htable tr:last-child td { border-bottom:none; }

/* SHAP bar */
.shap-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:11px; }
.shap-name{ color:#9ba8cc; width:200px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex-shrink:0; }
.shap-bar-wrap{ flex:1; height:8px; background:#131929; border-radius:99px; overflow:hidden; }
.shap-bar-pos{ height:100%; background:#f87171; border-radius:99px; }
.shap-bar-neg{ height:100%; background:#34d399; border-radius:99px; margin-left:auto; }
.shap-val{ color:#4a5580; width:44px; text-align:right; font-family:'IBM Plex Mono',monospace; }

/* Batch table */
.batch-tbl { font-size:12px; }

/* Inputs */
label { color:#6b7fa8 !important; font-size:12px !important; font-weight:500 !important; }
.stNumberInput input { font-family:'IBM Plex Mono',monospace !important; }
div[data-testid="stMetricValue"] { font-family:'Syne',sans-serif; font-size:1.5rem !important; color:#e2e8ff !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#0c0f1a; border-bottom:1px solid #1a2240; gap:0; }
.stTabs [data-baseweb="tab"]      { color:#3d5099; font-size:13px; font-weight:500; padding:0.6rem 1.2rem; border-radius:0; }
.stTabs [aria-selected="true"]    { color:#7c9dff !important; border-bottom:2px solid #7c9dff !important; background:transparent !important; }

/* Button */
.stButton>button {
    width:100%; height:3em; font-size:14px; font-weight:600;
    font-family:'IBM Plex Sans',sans-serif; letter-spacing:0.05em;
    background:linear-gradient(135deg,#2a4bff,#5577ff);
    color:#fff; border:none; border-radius:10px;
    transition:opacity .2s,transform .15s;
}
.stButton>button:hover  { opacity:.85; transform:translateY(-1px); }
.stButton>button:active { transform:scale(.98); }

/* Download btn override */
.stDownloadButton>button {
    background:#131929 !important; color:#7c9dff !important;
    border:1px solid #1a2240 !important; font-size:13px !important;
    height:2.4em !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem;'>
        <div class='ari-logo'>⬡ ARI</div>
        <div class='ari-tagline'>Audit Risk Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Session Stats")
    hist = st.session_state.history
    total = len(hist)
    flagged = sum(1 for h in hist if h["prob"] >= 0.5)

    c1, c2 = st.columns(2)
    c1.metric("Scored", total)
    c2.metric("Flagged", flagged)

    if total:
        avg = np.mean([h["prob"] for h in hist]) * 100
        st.metric("Avg Risk", f"{avg:.1f}%")

        # Mini sparkline using st.line_chart
        st.markdown("<div style='margin-top:0.5rem;font-size:10px;color:#3d5099;letter-spacing:0.1em;text-transform:uppercase;'>Session Risk Trend</div>", unsafe_allow_html=True)
        spark_df = pd.DataFrame({"Risk %": [h["prob"] * 100 for h in hist]})
        st.line_chart(spark_df, height=80, use_container_width=True)

    st.divider()
    st.markdown("#### Thresholds")
    st.caption(f"Low < {LOW_RISK_THRESHOLD*100:.0f}% · High ≥ {HIGH_RISK_THRESHOLD*100:.0f}%")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                "⬇ Download Audit Log",
                f,
                file_name="audit_log.jsonl",
                mime="application/jsonl",
            )

# ─────────────────────────────────────────────
# MAIN AREA — TABS
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.5rem;margin-top:1rem;'>
    <div class='page-title'>Audit Risk Detection</div>
    <div class='page-sub'>AI-powered transaction risk scoring · XGBoost model</div>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch, tab_history = st.tabs([
    "  🔍 Single Transaction  ",
    "  📂 Batch Upload  ",
    "  🕒 Session History  ",
])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE TRANSACTION
# ══════════════════════════════════════════════
with tab_single:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-label">Core Details</div>', unsafe_allow_html=True)
        amount      = st.number_input("Transaction Amount (₹)", min_value=0.01, max_value=10_000_000.0, value=10_000.0, step=100.0, format="%.2f")
        vendor_risk = st.slider("Vendor Risk Score", 0.0, 1.0, 0.50, step=0.01)
        col_a, col_b = st.columns(2)
        with col_a:
            prev_fraud = st.radio("Prior Fraud History?",   ["No", "Yes"], horizontal=True)
        with col_b:
            unusual    = st.radio("Unusual Timing?",         ["No", "Yes"], horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("⚙️ Advanced Options"):
            col_c, col_d, col_e = st.columns(3)
            with col_c:
                freq       = st.number_input("Frequency", 1, 100, 5)
            with col_d:
                round_amt  = st.radio("Round Amount?",   ["No", "Yes"], horizontal=True)
            with col_e:
                weekend    = st.radio("Weekend?",        ["No", "Yes"], horizontal=True)

        if amount <= 0:
            st.warning("Enter a transaction amount > ₹0 to continue.")
            st.stop()

        analyse = st.button("🔍 Analyse Transaction Risk")

    # ── Right panel: result ──
    with right:
        if analyse:
            prev_fraud_val = 1 if prev_fraud == "Yes" else 0
            unusual_val    = 1 if unusual    == "Yes" else 0
            round_amt_val  = 1 if round_amt  == "Yes" else 0
            weekend_val    = 1 if weekend    == "Yes" else 0

            with st.spinner("Running model…"):
                idf  = build_input_df(amount, vendor_risk, prev_fraud_val, unusual_val, freq, round_amt_val, weekend_val)
                prob = predict_risk(idf)
                time.sleep(0.3)  # let spinner show briefly

            tier_label, tier_key, bar_color = risk_tier(prob)
            signals = gather_signals(amount, vendor_risk, prev_fraud_val, unusual_val, round_amt_val, weekend_val)
            pct = prob * 100

            # Append to session history
            st.session_state.history.append({
                "time":   datetime.datetime.now().strftime("%H:%M:%S"),
                "amount": amount,
                "prob":   prob,
                "tier":   tier_label,
            })

            # Log to file
            log_prediction({
                "amount": amount, "vendor_risk": vendor_risk,
                "prev_fraud": prev_fraud_val, "unusual": unusual_val,
                "risk_score": round(prob, 4), "tier": tier_label,
            })

            # Score card
            st.markdown(f"""
            <div style="background:#0c0f1a;border:1px solid #2a3560;border-radius:14px;padding:2rem 1.5rem 1.5rem;text-align:center;margin-bottom:1rem;">
              <div style="font-size:10px;font-weight:600;letter-spacing:0.16em;text-transform:uppercase;color:#3d5099;margin-bottom:0.85rem;">Risk Score</div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:3.8rem;line-height:1;letter-spacing:-0.03em;color:{bar_color};">
                {pct:.1f}<span style="font-size:1.6rem;font-family:'Syne',sans-serif;">%</span>
              </div>
              <div style="height:6px;background:#131929;border-radius:99px;margin:0.9rem 0 0.75rem;overflow:hidden;">
                <div style="height:100%;width:{pct:.1f}%;background:{bar_color};border-radius:99px;"></div>
              </div>
              <span class="badge badge-{tier_key}">{tier_label}</span>
            </div>
            """, unsafe_allow_html=True)

            # Recommendation
            if prob >= 0.5:
                st.markdown('<div class="rec rec-audit"><span style="font-size:15px;">&#9888;</span> &nbsp;Requires manual audit review — escalate to compliance.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="rec rec-clear"><span style="font-size:15px;">&#10003;</span> &nbsp;No immediate action required.</div>', unsafe_allow_html=True)

            # Signals
            if signals:
                pills = " ".join(f'<span class="sig-pill">⚡ {s}</span>' for s in signals)
                st.markdown(f'<div style="margin-top:0.75rem;display:flex;flex-wrap:wrap;gap:6px;">{pills}</div>', unsafe_allow_html=True)

            # SHAP explainability
            if SHAP_AVAILABLE:
                st.markdown('<div style="background:#0c0f1a;border:1px solid #1a2240;border-radius:14px;padding:1.25rem 1.5rem;margin-top:0.75rem;"><div style="font-size:10px;font-weight:600;letter-spacing:0.16em;text-transform:uppercase;color:#3d5099;margin-bottom:0.85rem;">Feature Contributions (SHAP)</div>', unsafe_allow_html=True)
                shap_vals, feat_names = compute_shap(idf)
                if shap_vals is not None:
                    pairs = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:8]
                    max_abs = max(abs(v) for _, v in pairs) or 1
                    for fname, fval in pairs:
                        pct_bar = abs(fval) / max_abs * 100
                        direction = "pos" if fval > 0 else "neg"
                        bar_html = f'<div class="shap-bar-{direction}" style="width:{pct_bar:.1f}%;"></div>'
                        st.markdown(f"""
                        <div class="shap-row">
                          <div class="shap-name">{fname}</div>
                          <div class="shap-bar-wrap">{bar_html}</div>
                          <div class="shap-val">{"+" if fval>0 else ""}{fval:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('<div style="font-size:10px;color:#3d5099;margin-top:6px;">Red = increases risk &nbsp;·&nbsp; Green = decreases risk</div>', unsafe_allow_html=True)
                else:
                    st.caption("SHAP explanation unavailable for this prediction.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#0c0f1a;border:1px dashed #1a2240;border-radius:14px;padding:1rem 1.5rem;margin-top:0.75rem;font-size:12px;color:#3d5099;">
                  Run <code style="background:#131929;padding:1px 6px;border-radius:4px;color:#7c9dff;">pip install shap</code> and restart to enable per-prediction feature explanations.
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="min-height:260px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:0.5rem;text-align:center;">
              <div style="font-size:2.5rem;">🔍</div>
              <div style="color:#3d5099;font-size:13px;">Fill in transaction details<br>and click Analyse to see the risk score.</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown("""
    <div class="card">
      <div class="card-label">Batch CSV Scoring</div>
      <div style="font-size:13px;color:#6b7fa8;line-height:1.6;">
        Upload a CSV with columns: <code>Transaction_Amount</code>, <code>Vendor_Risk_Score</code>,
        <code>Previous_Fraud_Flag</code>, <code>Unusual_Time_Flag</code>.
        Optional: <code>Transaction_Frequency</code>, <code>Round_Amount_Flag</code>, <code>Weekend_Transaction</code>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            missing = [c for c in BATCH_REQUIRED_COLS if c not in raw.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                with st.spinner(f"Scoring {len(raw)} transactions…"):
                    scored = batch_score(raw)

                st.success(f"✅ Scored {len(scored)} transactions.")

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total",        len(scored))
                m2.metric("Low Risk",     (scored["Risk_Tier"] == "Low Risk").sum())
                m3.metric("Medium Risk",  (scored["Risk_Tier"] == "Medium Risk").sum())
                m4.metric("High Risk",    (scored["Risk_Tier"] == "High Risk").sum())

                # Colour-coded results table
                def colour_tier(val):
                    colours = {"Low Risk": "#34d399", "Medium Risk": "#fbbf24", "High Risk": "#f87171"}
                    return f"color: {colours.get(val, '#9ba8cc')}"

                st.dataframe(
                    scored.style.applymap(colour_tier, subset=["Risk_Tier"]),
                    use_container_width=True,
                    height=360,
                )

                csv_out = scored.to_csv(index=False).encode()
                st.download_button(
                    "⬇ Download Scored CSV",
                    csv_out,
                    file_name="audit_risk_scored.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        # Sample template download
        sample = pd.DataFrame([
            {"Transaction_Amount": 15000, "Vendor_Risk_Score": 0.3, "Previous_Fraud_Flag": 0, "Unusual_Time_Flag": 0},
            {"Transaction_Amount": 95000, "Vendor_Risk_Score": 0.85,"Previous_Fraud_Flag": 1, "Unusual_Time_Flag": 1},
            {"Transaction_Amount": 4200,  "Vendor_Risk_Score": 0.5, "Previous_Fraud_Flag": 0, "Unusual_Time_Flag": 0},
        ])
        st.download_button(
            "⬇ Download Sample Template",
            sample.to_csv(index=False).encode(),
            file_name="sample_transactions.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════
# TAB 3 — SESSION HISTORY
# ══════════════════════════════════════════════
with tab_history:
    if not st.session_state.history:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;color:#3d5099;font-size:13px;">
          No transactions scored yet this session.<br>Use the Single Transaction tab to get started.
        </div>
        """, unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        # Trend chart
        st.markdown('<div class="card"><div class="card-label">Risk Score Trend</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({"Risk %": [h["prob"] * 100 for h in st.session_state.history]})
        st.line_chart(chart_df, height=160, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Table
        st.markdown('<div class="card"><div class="card-label">Transaction Log</div>', unsafe_allow_html=True)

        rows_html = ""
        for h in reversed(st.session_state.history):
            tier_colors = {"Low Risk": "#34d399", "Medium Risk": "#fbbf24", "High Risk": "#f87171"}
            tc = tier_colors.get(h["tier"], "#9ba8cc")
            rows_html += f"""
            <tr>
              <td>{h['time']}</td>
              <td style="font-family:'IBM Plex Mono',monospace;">₹{h['amount']:,.0f}</td>
              <td style="font-family:'IBM Plex Mono',monospace;color:{tc};">{h['prob']*100:.1f}%</td>
              <td style="color:{tc};font-weight:600;">{h['tier']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="htable">
          <thead><tr><th>Time</th><th>Amount</th><th>Score</th><th>Tier</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🗑 Clear Session History"):
            st.session_state.history = []
            st.rerun()