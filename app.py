import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data import load_portfolio
from valuation import (full_valuation, build_discount_rate,
                       dcf_valuation, monte_carlo_valuation)
from narrative import generate_credit_narrative
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Private Credit Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  h1 { font-size: 1.7rem !important; font-weight: 700 !important; margin-bottom: 0 !important; }
  h2 { font-size: 1.2rem !important; font-weight: 600 !important; }
  h3 { font-size: 1rem !important; font-weight: 600 !important; }
  [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; }
  [data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #6B7280 !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-divider { border: none; border-top: 1px solid #E5E7EB; margin: 1rem 0; }
  .memo-section { border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; line-height: 1.85; font-size: 0.88rem; }
  .memo-label { font-size: 0.65rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; }
  .memo-health { background: #EFF6FF; border-left: 5px solid #3B82F6; }
  .memo-health .memo-label { color: #1D4ED8; }
  .memo-valuation { background: #F0FDF4; border-left: 5px solid #22C55E; }
  .memo-valuation .memo-label { color: #15803D; }
  .memo-risks { background: #FFFBEB; border-left: 5px solid #F59E0B; }
  .memo-risks .memo-label { color: #B45309; }
  .memo-action { background: #FAF5FF; border-left: 5px solid #8B5CF6; }
  .memo-action .memo-label { color: #6D28D9; }
  .status-pill { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; }
  .pill-red    { background: #FEE2E2; color: #991B1B; }
  .pill-amber  { background: #FEF3C7; color: #92400E; }
  .pill-green  { background: #DCFCE7; color: #166534; }
  .sidebar-meta { font-size: 0.72rem; color: #9CA3AF; line-height: 1.8; }
  .section-chip { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #6B7280; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── DATA & HELPERS ────────────────────────────────────────────────
df = load_portfolio()
RADAR_CATS = ["Leverage","Coverage","LTV","Sentiment","Risk Flags","Yield"]

def radar_vals(row):
    return [
        min(row["net_leverage"] / 6.0, 1.0),
        max(0, 1 - (row["interest_coverage"] - 1) / 4.0),
        row["ltv_pct"] / 100,
        max(0, (-row["sentiment_score"] + 1) / 2),
        row["risk_flags"] / 4.0,
        min(row["coupon_pct"] / 15.0, 1.0),
    ]

def status_pill(flags):
    if flags >= 3: return "HIGH RISK", "pill-red"
    if flags >= 1: return "WATCH", "pill-amber"
    return "HEALTHY", "pill-green"

def render_memo(memo_text):
    section_map = {
        "HEALTH SUMMARY":       ("memo-health",     "Health Summary"),
        "VALUATION ASSESSMENT": ("memo-valuation",  "Valuation Assessment"),
        "KEY RISKS":            ("memo-risks",       "Key Risks"),
        "RECOMMENDED ACTION":   ("memo-action",      "Recommended Action"),
    }
    current_key  = None
    current_body = []

    def flush(key, lines):
        if not key or not lines: return
        cls, label = section_map[key]
        body = " ".join(lines).replace("**","").replace("##","").strip()
        st.markdown(
            f'<div class="memo-section {cls}">'
            f'<div class="memo-label">{label}</div>'
            f'<div style="color:#1F2937;">{body}</div></div>',
            unsafe_allow_html=True
        )

    for line in memo_text.split("\n"):
        matched = False
        for key in section_map:
            if key in line.upper():
                flush(current_key, current_body)
                current_key  = key
                current_body = []
                matched = True
                break
        if not matched and line.strip():
            current_body.append(line.strip())
    flush(current_key, current_body)

def kpi_row(metrics: list):
    cols = st.columns(len(metrics))
    for col, (label, value, delta, inv) in zip(cols, metrics):
        if delta:
            col.metric(label, value, delta=delta,
                       delta_color="inverse" if inv else "normal")
        else:
            col.metric(label, value)

PALETTE = px.colors.qualitative.Set2

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Credit Intelligence")
    st.markdown('<div class="sidebar-meta">Private Credit · AI-Powered<br>ASC 820 / IFRS 13 Compliant</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Portfolio Overview",
        "⚠️  Risk Watchlist",
        "📐  Valuation Engine",
        "🤖  Credit Memo",
        "📈  NAV Scenarios",
        "💬  Ask the AI",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="section-chip">Methodology</div>', unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-meta">
    Discount rate: Build-up method<br>
    DCF 50% · Comps 30% · MC P50 20%<br>
    Monte Carlo: 10,000 simulations<br>
    Credit score: Adapted Altman Z<br>
    Framework: ASC 820 / IFRS 13
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Portfolio Overview")
    st.caption(f"{len(df)} loans · ${df['principal_mm'].sum():.0f}M total exposure · "
               f"{int((df['risk_flags']>0).sum())} flagged · "
               f"{int(df['covenant_breached'].sum())} covenant breaches")
    st.markdown('<hr class="kpi-divider">', unsafe_allow_html=True)

    kpi_row([
        ("Total Loans",       str(len(df)),                             None, False),
        ("Total Exposure",    f"${df['principal_mm'].sum():.0f}M",      None, False),
        ("Loans at Risk",     str(int((df['risk_flags']>0).sum())),     None, False),
        ("Avg Coupon",        f"{df['coupon_pct'].mean():.1f}%",        None, False),
        ("Covenant Breaches", str(int(df['covenant_breached'].sum())),  None, False),
    ])

    st.markdown("---")
    st.markdown('<div class="section-chip">Portfolio Risk Heatmap</div>', unsafe_allow_html=True)
    st.caption("Each cell 0–1 · green = low risk · red = high risk")

    heat_rows = []
    for _, row in df.iterrows():
        heat_rows.append({
            "Borrower":   row["borrower"],
            "Leverage":   round(min(row["net_leverage"]/6.0, 1.0), 2),
            "Coverage":   round(max(0,1-(row["interest_coverage"]-1)/4), 2),
            "LTV":        round(row["ltv_pct"]/100, 2),
            "Sentiment":  round(max(0,(-row["sentiment_score"]+1)/2), 2),
            "Risk Flags": round(row["risk_flags"]/4.0, 2),
            "Yield":      round(min(row["coupon_pct"]/15.0, 1.0), 2),
        })
    heat_df = pd.DataFrame(heat_rows).set_index("Borrower")
    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=["#4ADE80","#FCD34D","#F87171"],
        zmin=0, zmax=1, text_auto=".2f", aspect="auto"
    )
    fig_heat.update_layout(height=360, margin=dict(t=20,b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-chip">Exposure by Borrower</div>', unsafe_allow_html=True)
        fig_exp = px.bar(df, x="borrower", y="principal_mm",
                         color="sector",
                         labels={"principal_mm":"Principal ($M)","borrower":""},
                         color_discrete_sequence=PALETTE)
        fig_exp.update_layout(height=300, showlegend=True,
                               xaxis_tickangle=-30, margin=dict(t=10))
        st.plotly_chart(fig_exp, use_container_width=True)

    with col2:
        st.markdown('<div class="section-chip">Leverage vs Coverage Map</div>', unsafe_allow_html=True)
        fig_lc = px.scatter(df, x="net_leverage", y="interest_coverage",
                            color="sector", size="principal_mm",
                            hover_name="borrower",
                            labels={"net_leverage":"Net Leverage (x)",
                                    "interest_coverage":"Interest Coverage (x)"},
                            color_discrete_sequence=PALETTE)
        fig_lc.add_hline(y=2.0, line_dash="dash", line_color="#EF4444",
                         annotation_text="Coverage floor 2x",
                         annotation_font_size=11)
        fig_lc.add_vline(x=5.0, line_dash="dash", line_color="#EF4444",
                         annotation_text="Leverage limit 5x",
                         annotation_font_size=11)
        fig_lc.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig_lc, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-chip">Borrower Risk Radar — Side by Side</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    b1 = col1.selectbox("Borrower A", df["borrower"].tolist(), index=1, key="r1")
    b2 = col2.selectbox("Borrower B", df["borrower"].tolist(), index=4, key="r2")
    fig_r = go.Figure()
    for bname, colour in [(b1,"#3B82F6"),(b2,"#F97316")]:
        row  = df[df["borrower"]==bname].iloc[0]
        vals = radar_vals(row)+[radar_vals(row)[0]]
        cats = RADAR_CATS+[RADAR_CATS[0]]
        fig_r.add_trace(go.Scatterpolar(r=vals, theta=cats,
                        fill="toself", name=bname,
                        line_color=colour, opacity=0.7))
    fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                        showlegend=True, height=380,
                        title="Higher score = higher risk on that dimension")
    st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — RISK WATCHLIST
# ══════════════════════════════════════════════════════════════════
elif "Watchlist" in page:
    st.title("Risk Watchlist")
    st.markdown('<hr class="kpi-divider">', unsafe_allow_html=True)

    kpi_row([
        ("Flagged Loans",     str(int((df['risk_flags']>0).sum())),    None, False),
        ("Covenant Breaches", str(int(df['covenant_breached'].sum())), None, False),
        ("Low Coverage",      str(int(df['low_coverage'].sum())),      None, False),
        ("High LTV (>75%)",   str(int(df['high_ltv'].sum())),          None, False),
    ])

    st.markdown("---")
    watchlist = df[df["risk_flags"]>0].sort_values("risk_flags",
                                                    ascending=False)
    if watchlist.empty:
        st.success("No loans currently flagged.")
    else:
        for _, row in watchlist.iterrows():
            label, pill_cls = status_pill(int(row["risk_flags"]))
            header = (
                f'<span class="status-pill {pill_cls}">{label}</span>'
                f'&nbsp;&nbsp;<b>{row["borrower"]}</b>'
                f' — {row["sector"]} · '
                f'${row["principal_mm"]}M · '
                f'{int(row["risk_flags"])}/4 flags'
            )
            with st.expander(
                f'{label}  {row["borrower"]} — '
                f'{row["sector"]} · '
                f'${row["principal_mm"]}M · '
                f'{int(row["risk_flags"])}/4 flags'
            ):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Net Leverage",
                          f"{row['net_leverage']}x",
                          delta=f"limit {row['leverage_limit']}x",
                          delta_color="inverse")
                c2.metric("Coverage",  f"{row['interest_coverage']}x")
                c3.metric("LTV",       f"{row['ltv_pct']}%")
                c4.metric("Sentiment", f"{row['sentiment_score']}")

                triggered = []
                if row["covenant_breached"]: triggered.append("⚠️ Covenant breached")
                if row["low_coverage"]:      triggered.append("📉 Low interest coverage")
                if row["high_ltv"]:          triggered.append("🏦 High LTV")
                if row["neg_sentiment"]:     triggered.append("📰 Negative sentiment")
                st.error("  ·  ".join(triggered))
                st.caption(f"📰 {row['recent_news']}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-chip">Risk Flags by Borrower</div>', unsafe_allow_html=True)
            fig_f = px.bar(watchlist, x="borrower", y="risk_flags",
                           color="risk_flags",
                           color_continuous_scale=["#FCD34D","#F87171","#B91C1C"],
                           labels={"risk_flags":"Flags","borrower":""})
            fig_f.update_layout(height=280, showlegend=False,
                                 xaxis_tickangle=-20, margin=dict(t=10))
            st.plotly_chart(fig_f, use_container_width=True)

        with col2:
            st.markdown('<div class="section-chip">Stressed Borrower Profiles</div>', unsafe_allow_html=True)
            fig_rr = go.Figure()
            cols = ["#EF4444","#F97316","#EAB308","#22C55E","#3B82F6"]
            for i,(_, row) in enumerate(watchlist.iterrows()):
                vals = radar_vals(row)+[radar_vals(row)[0]]
                cats = RADAR_CATS+[RADAR_CATS[0]]
                fig_rr.add_trace(go.Scatterpolar(r=vals, theta=cats,
                    fill="toself", name=row["borrower"],
                    line_color=cols[i%len(cols)], opacity=0.65))
            fig_rr.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                                  height=280, margin=dict(t=10))
            st.plotly_chart(fig_rr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — VALUATION ENGINE
# ══════════════════════════════════════════════════════════════════
elif "Valuation" in page:
    st.title("Valuation Engine")
    st.caption("ASC 820 / IFRS 13 · Build-up discount rate · DCF · EV/EBITDA comps · Monte Carlo")
    st.markdown('<hr class="kpi-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    borrower   = col1.selectbox("Select borrower",
                                df["borrower"].tolist(), key="ve_b")
    instrument = col2.selectbox("Instrument type",[
        "direct_lending","unitranche","second_lien",
        "mezzanine","pik","asset_based"])
    row  = df[df["borrower"]==borrower].iloc[0]
    base = build_discount_rate(row, instrument)

    st.markdown("---")
    st.markdown('<div class="section-chip">Interactive Discount Rate Builder — drag sliders to update DCF NAV in real time</div>', unsafe_allow_html=True)

    sc1,sc2,sc3 = st.columns(3)
    sofr = sc1.slider("SOFR base (%)",        2.0,  8.0, float(base["sofr_base"]),        0.05)
    csp  = sc2.slider("Credit spread (%)",    0.5, 10.0, float(base["credit_spread"]),    0.05)
    ill  = sc3.slider("Illiquidity prem (%)", 0.5,  4.0, float(base["illiquidity_prem"]), 0.05)
    sc4,sc5,sc6 = st.columns(3)
    cxp  = sc4.slider("Complexity prem (%)",  0.0,  2.0, float(base["complexity_prem"]),  0.05)
    sec  = sc5.slider("Sector adj (%)",       -0.5, 1.5, float(base["sector_adj"]),       0.05)
    cov  = sc6.slider("Covenant adj (%)",     0.0,  2.0, float(base["covenant_adj"]),     0.05)

    live_rate = round(sofr+csp+ill+cxp+sec+cov, 2)
    live_dcf  = dcf_valuation(row, live_rate)

    kpi_row([
        ("Live Discount Rate", f"{live_rate:.2f}%",                                   None,  False),
        ("Live DCF NAV",       f"${live_dcf['nav']:.2f}M",
         f"{live_dcf['nav_to_par']:.1f}% of par",                                             False),
        ("Duration",           f"{live_dcf['duration_yrs']:.2f} yr",                  None,  False),
        ("DV01",               f"${live_dcf['dv01_mm']:.3f}M",
         "per 100bps move",                                                                    False),
    ])

    names = ["SOFR","Credit spread","Illiquidity","Complexity","Sector adj","Covenant adj"]
    vals  = [sofr, csp, ill, cxp, sec, cov]
    fig_wf = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=["#3B82F6","#F59E0B","#10B981","#8B5CF6","#EF4444","#DC2626"],
        text=[f"{v:.2f}%" for v in vals], textposition="outside"
    ))
    fig_wf.add_hline(y=live_rate, line_dash="dash", line_color="#EF4444",
                     annotation_text=f"Total: {live_rate:.2f}%",
                     annotation_font_size=12)
    fig_wf.update_layout(title="Discount Rate Components",
                          yaxis_title="Rate (%)", showlegend=False,
                          height=300, margin=dict(t=40,b=20))
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("---")
    if st.button("▶  Run Full Valuation  (DCF · Comps · Monte Carlo · Z-score)",
                 type="primary"):
        with st.spinner("Running valuation engine..."):
            val = full_valuation(row, instrument)
        st.session_state[f"val_{borrower}"] = val

    if f"val_{borrower}" in st.session_state:
        val = st.session_state[f"val_{borrower}"]
        r,d,c,m,z = (val["rate_components"], val["dcf"],
                      val["comps"], val["monte_carlo"],
                      val["credit_score"])

        nav_col = ("#4ADE80" if val["nav_to_par_pct"]>=98
                   else "#FCD34D" if val["nav_to_par_pct"]>=90
                   else "#F87171")
        st.markdown(
            f'<div style="background:{nav_col}22;border-left:5px solid {nav_col};'
            f'border-radius:6px;padding:12px 18px;margin:1rem 0;">'
            f'<span style="font-weight:700;font-size:1rem;">'
            f'{val["nav_flag"]}</span>'
            f'<span style="color:#6B7280;font-size:0.85rem;"> · '
            f'Blended NAV <b>${val["blended_nav_mm"]}M</b> '
            f'({val["nav_to_par_pct"]}% of par) · '
            f'{val["asc820_level"]} · {val["weighting"]}</span></div>',
            unsafe_allow_html=True
        )

        kpi_row([
            ("Blended NAV",    f"${val['blended_nav_mm']}M",
             f"{val['nav_to_par_pct']}% of par",                     False),
            ("Discount Rate",  f"{r['total_rate_pct']}%",
             f"BSL premium {r['bsl_premium_pct']}%",                 False),
            ("Altman Z-Score", f"{z['z_score']}",
             z['zone'],                                               False),
            ("Default Prob",   f"{m['pd_applied']}%",
             f"Exp. loss ${m['expected_loss_mm']}M",                  True),
        ])

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-chip">DCF Cash Flow Analysis</div>', unsafe_allow_html=True)
            cfs = d["cash_flows"]
            fig_dcf = go.Figure()
            fig_dcf.add_bar(x=[f"Y{c['period']}" for c in cfs],
                            y=[c["cash_flow"]     for c in cfs],
                            name="Cash Flow", marker_color="#3B82F6")
            fig_dcf.add_bar(x=[f"Y{c['period']}" for c in cfs],
                            y=[c["present_value"] for c in cfs],
                            name="Present Value", marker_color="#F59E0B")
            fig_dcf.update_layout(barmode="group", height=280,
                                   title=f"DCF NAV ${d['nav']}M · {d['nav_to_par']}% of par",
                                   margin=dict(t=40,b=10))
            st.plotly_chart(fig_dcf, use_container_width=True)
            st.caption(f"Duration {d['duration_yrs']}yr · DV01 ${d['dv01_mm']}M/100bps")

        with col2:
            st.markdown('<div class="section-chip">Monte Carlo NAV Distribution</div>', unsafe_allow_html=True)
            stress = st.toggle("Stress — double default probability")
            if stress:
                with st.spinner("Stress scenario..."):
                    rs = row.copy()
                    rs["risk_flags"] = min(int(row["risk_flags"])+2, 4)
                    mc_s = monte_carlo_valuation(rs, instrument)
                navs = np.array(mc_s["nav_distribution"])
                p10,p50,p90 = mc_s["p10"],mc_s["p50"],mc_s["p90"]
                bar_c, pd_lbl = "#F87171", f"Stress PD: {mc_s['pd_applied']}%"
            else:
                navs = np.array(m["nav_distribution"])
                p10,p50,p90 = m["p10"],m["p50"],m["p90"]
                bar_c, pd_lbl = "#4ADE80", f"Base PD: {m['pd_applied']}%"

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=navs, nbinsx=60,
                             marker_color=bar_c, opacity=0.8,
                             name="NAV simulations"))
            for lbl,vp,col in [("P10",p10,"#EF4444"),
                                ("P50",p50,"#F59E0B"),
                                ("P90",p90,"#22C55E")]:
                fig_mc.add_vline(x=vp, line_dash="dash",
                                 line_color=col,
                                 annotation_text=f"{lbl} ${vp}M",
                                 annotation_font_size=11)
            fig_mc.update_layout(
                title=f"10,000 simulations · {pd_lbl}",
                xaxis_title="NAV ($M)", yaxis_title="Frequency",
                height=280, margin=dict(t=40,b=10))
            st.plotly_chart(fig_mc, use_container_width=True)
            st.caption(f"P10 ${p10}M · P50 ${p50}M · P90 ${p90}M · "
                       f"E[Loss] ${m['expected_loss_mm']}M")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-chip">Comparable Company Analysis (EV/EBITDA)</div>', unsafe_allow_html=True)
            comps_data = pd.DataFrame([
                ["Sector Multiple", f"{c['sector_multiple']}x EV/EBITDA"],
                ["Implied EV",      f"${c['implied_ev']}M"],
                ["Debt Coverage",   f"{c['ev_coverage']}x"],
                ["Equity Cushion",  f"${c['equity_cushion_mm']}M ({c['equity_cushion_pct']}%)"],
                ["Bear Case EV",    f"${c['ev_bear_case']}M"],
                ["Bull Case EV",    f"${c['ev_bull_case']}M"],
            ], columns=["Metric","Value"])
            st.dataframe(comps_data, hide_index=True,
                         use_container_width=True)
            st.caption(c["coverage_assessment"])

        with col2:
            st.markdown('<div class="section-chip">Altman Z-Score Component Breakdown</div>', unsafe_allow_html=True)
            zc = z["components"]
            fig_z = go.Figure(go.Bar(
                x=list(zc.keys()), y=list(zc.values()),
                marker_color=["#3B82F6","#10B981","#F59E0B",
                              "#8B5CF6","#EF4444"],
                text=[f"{v:.3f}" for v in zc.values()],
                textposition="outside"
            ))
            fig_z.update_layout(
                title=f"Z = {z['z_score']} — {z['zone']}",
                yaxis_title="Score", height=280,
                margin=dict(t=40,b=10))
            st.plotly_chart(fig_z, use_container_width=True)

        if val.get("recovery"):
            st.markdown("---")
            st.error("⚠️ Covenant Breach — Recovery Analysis Active")
            rec = val["recovery"]
            kpi_row([
                ("Recovery Rate",   f"{rec['recovery_rate']*100:.1f}%",  None, False),
                ("Recovery Value",  f"${rec['recovery_value_mm']}M",     None, False),
                ("Loss Given Default", f"{rec['loss_given_default']*100:.1f}%", None, False),
                ("Asset Coverage",  f"{rec['asset_coverage_ratio']}x",  None, False),
            ])


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — CREDIT MEMO
# ══════════════════════════════════════════════════════════════════
elif "Memo" in page:
    st.title("AI Credit Memo Generator")
    st.caption("Institutional-grade narratives grounded in full valuation stack")
    st.markdown('<hr class="kpi-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    borrower   = col1.selectbox("Select borrower",
                                df["borrower"].tolist(), key="cm_b")
    instrument = col2.selectbox("Instrument type",[
        "direct_lending","unitranche","second_lien",
        "mezzanine","pik","asset_based"], key="cm_i")
    row   = df[df["borrower"]==borrower].iloc[0]
    flags = int(row["risk_flags"])

    kpi_row([
        ("Net Leverage",  f"{row['net_leverage']}x",
         f"limit {row['leverage_limit']}x",        True),
        ("Coverage",      f"{row['interest_coverage']}x", None, False),
        ("LTV",           f"{row['ltv_pct']}%",           None, False),
        ("Sentiment",     f"{row['sentiment_score']}",    None, False),
        ("Risk Flags",    f"{flags} / 4",                 None, False),
    ])

    label, pill_cls = status_pill(flags)
    st.markdown(
        f'<div style="margin:0.5rem 0 1rem;">'
        f'<span class="status-pill {pill_cls}">{label}</span>'
        f'<span style="color:#6B7280;font-size:0.82rem;margin-left:10px;">'
        f'{row["recent_news"]}</span></div>',
        unsafe_allow_html=True
    )

    if st.button("⚡  Generate Credit Memo", type="primary"):
        with st.spinner("Running full valuation · generating institutional memo..."):
            val  = full_valuation(row, instrument)
            memo = generate_credit_narrative(row, val)
        st.session_state[f"memo_{borrower}"]    = memo
        st.session_state[f"memoval_{borrower}"] = val

    if f"memo_{borrower}" in st.session_state:
        memo = st.session_state[f"memo_{borrower}"]
        val  = st.session_state[f"memoval_{borrower}"]

        st.markdown("---")

        # Valuation summary strip above memo
        v_col = ("#4ADE80" if val["nav_to_par_pct"]>=98
                 else "#FCD34D" if val["nav_to_par_pct"]>=90
                 else "#F87171")
        st.markdown(
            f'<div style="background:{v_col}18;border:1px solid {v_col}66;'
            f'border-radius:8px;padding:10px 16px;margin-bottom:1rem;'
            f'font-size:0.82rem;color:#374151;">'
            f'<b>Valuation summary</b> · '
            f'Blended NAV <b>${val["blended_nav_mm"]}M</b> '
            f'({val["nav_to_par_pct"]}% of par) · '
            f'Discount rate <b>{val["rate_components"]["total_rate_pct"]}%</b> · '
            f'Z-score <b>{val["credit_score"]["z_score"]}</b> '
            f'({val["credit_score"]["zone"]}) · '
            f'Default prob <b>{val["monte_carlo"]["pd_applied"]}%</b> · '
            f'{val["asc820_level"]} · {val["valuation_date"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Render memo immediately — colour-coded sections
        render_memo(memo)

        # Download button below memo
        st.markdown("---")
        col1, col2 = st.columns([3,1])
        col1.caption(f"Generated: {val['valuation_date']} · "
                     f"{val['methodology_note']}")
        col2.download_button(
            "📥 Download memo",
            memo,
            file_name=f"{borrower}_credit_memo.txt",
            mime="text/plain"
        )


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — NAV SCENARIOS
# ══════════════════════════════════════════════════════════════════
elif "NAV" in page:
    st.title("NAV Scenario Analysis")
    st.caption("Real-time portfolio valuation as discount rate changes")
    st.markdown('<hr class="kpi-divider">', unsafe_allow_html=True)

    rate = st.slider("Discount Rate (%)", 4.0, 20.0, 10.0,
                     step=0.25, format="%.2f%%")

    def loan_nav(r, dr):
        d  = dr/100; p = r["principal_mm"]
        c  = p*r["coupon_pct"]/100
        cf = [c]*4; cf[-1]+=p
        return sum(v/(1+d)**t for t,v in enumerate(cf,1))

    df["s_nav"] = df.apply(lambda r: loan_nav(r, rate), axis=1)
    tot_nav = df["s_nav"].sum(); tot_par = df["principal_mm"].sum()

    kpi_row([
        ("Portfolio NAV", f"${tot_nav:.1f}M",                   None,  False),
        ("Par Value",     f"${tot_par:.0f}M",                   None,  False),
        ("NAV vs Par",    f"${tot_nav-tot_par:.1f}M",           None,  False),
        ("Avg NAV / Par", f"{tot_nav/tot_par*100:.1f}%",        None,  False),
    ])

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-chip">Loan NAV at Current Rate</div>', unsafe_allow_html=True)
        fig_b = px.bar(df, x="borrower", y="s_nav", color="sector",
                       labels={"s_nav":"NAV ($M)","borrower":""},
                       color_discrete_sequence=PALETTE)
        fig_b.update_layout(height=300, xaxis_tickangle=-30, margin=dict(t=10))
        st.plotly_chart(fig_b, use_container_width=True)

    with col2:
        st.markdown('<div class="section-chip">Portfolio NAV Sensitivity Curve</div>', unsafe_allow_html=True)
        rates = np.arange(4.0, 20.25, 0.25)
        navs  = [df.apply(lambda r: loan_nav(r,dr),axis=1).sum() for dr in rates]
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=rates, y=navs, mode="lines",
                        line=dict(color="#3B82F6",width=2.5),
                        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
        fig_c.add_vline(x=rate, line_dash="dash", line_color="#EF4444",
                        annotation_text=f"{rate:.2f}%", annotation_font_size=11)
        fig_c.add_hline(y=tot_par, line_dash="dot", line_color="#9CA3AF",
                        annotation_text="Par", annotation_font_size=11)
        fig_c.update_layout(xaxis_title="Discount Rate (%)",
                             yaxis_title="Portfolio NAV ($M)",
                             height=300, margin=dict(t=10))
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-chip">DV01 — Dollar Rate Sensitivity per Loan ($ change per 100bps increase)</div>', unsafe_allow_html=True)
    dv_rows = []
    for _, r in df.iterrows():
        nb = loan_nav(r, rate); nu = loan_nav(r, rate+1.0)
        dv_rows.append({
            "Borrower":    r["borrower"],
            "Sector":      r["sector"],
            "Principal":   f"${r['principal_mm']}M",
            "Current NAV": f"${nb:.2f}M",
            "NAV +100bps": f"${nu:.2f}M",
            "DV01 ($M)":   round(nb-nu, 3),
        })
    dv_df = pd.DataFrame(dv_rows).sort_values("DV01 ($M)", ascending=False)
    st.dataframe(dv_df, hide_index=True, use_container_width=True)

    fig_dv = px.bar(dv_df, x="Borrower", y="DV01 ($M)", color="Sector",
                    title="Most rate-sensitive loans first",
                    color_discrete_sequence=PALETTE)
    fig_dv.update_layout(height=280, xaxis_tickangle=-30, margin=dict(t=40))
    st.plotly_chart(fig_dv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 6 — ASK THE AI
# ══════════════════════════════════════════════════════════════════
elif "Ask" in page:
    st.title("Ask the AI")
    st.caption("Free-text Q&A grounded in live valuation data — "
               "the agentic layer 73Strings has on their roadmap but hasn't shipped")
    st.markdown('<hr class="kpi-divider">', unsafe_allow_html=True)

    borrower = st.selectbox("Borrower context",
                            ["All borrowers"]+df["borrower"].tolist(),
                            key="ai_b")

    if "ask_q" not in st.session_state:
        st.session_state["ask_q"] = ""

    st.markdown('<div class="section-chip">Suggested questions</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    if c1.button("Why is BetaCorp below par?"):
        st.session_state["ask_q"] = (
            "Why is BetaCorp trading at only 89.4% of par? "
            "Break down every contributing factor — discount rate "
            "components, covenant breach, Z-score, Monte Carlo default "
            "probability, and recovery risk. Reference specific numbers."
        )
    if c2.button("SOFR drops 100bps — impact?"):
        st.session_state["ask_q"] = (
            "If SOFR drops by exactly 100 basis points from 4.33% to 3.33%, "
            "calculate the new NAV for each borrower using DV01 and show "
            "total portfolio impact in dollars. Rank by sensitivity."
        )
    if c3.button("Worst recovery profile?"):
        st.session_state["ask_q"] = (
            "Which borrower has the worst recovery profile? Compare recovery "
            "rates, LGD, asset coverage, and default probability across all "
            "borrowers. Rank worst to best and explain why."
        )

    question = st.text_input("Your question",
                              value=st.session_state["ask_q"],
                              placeholder="e.g. Which loans are most exposed to a rate rise?")
    st.session_state["ask_q"] = question

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analysing portfolio data..."):
                if borrower == "All borrowers":
                    lines = []
                    for _, r in df.iterrows():
                        v = full_valuation(r,"direct_lending")
                        rec = (
                            f"Recovery ${v['recovery']['recovery_value_mm']}M "
                            f"({round(v['recovery']['recovery_rate']*100,1)}%), "
                            f"LGD {round(v['recovery']['loss_given_default']*100,1)}%"
                            if v.get("recovery") else "No breach"
                        )
                        lines.append(
                            f"\n{r['borrower']} ({r['sector']}):\n"
                            f"  NAV ${v['blended_nav_mm']}M "
                            f"({v['nav_to_par_pct']}% par) — {v['nav_flag']}\n"
                            f"  Rate {v['rate_components']['total_rate_pct']}% "
                            f"[SOFR {v['rate_components']['sofr_base']}% + "
                            f"spread {v['rate_components']['credit_spread']}% + "
                            f"illiq {v['rate_components']['illiquidity_prem']}% + "
                            f"cov {v['rate_components']['covenant_adj']}%]\n"
                            f"  DCF ${v['dcf']['nav']}M | "
                            f"DV01 ${v['dcf']['dv01_mm']}M | "
                            f"Dur {v['dcf']['duration_yrs']}yr\n"
                            f"  MC P10 ${v['monte_carlo']['p10']}M | "
                            f"P50 ${v['monte_carlo']['p50']}M | "
                            f"P90 ${v['monte_carlo']['p90']}M | "
                            f"PD {v['monte_carlo']['pd_applied']}% | "
                            f"E[Loss] ${v['monte_carlo']['expected_loss_mm']}M\n"
                            f"  Comps EV ${v['comps']['implied_ev']}M | "
                            f"Cov {v['comps']['ev_coverage']}x | "
                            f"Cushion ${v['comps']['equity_cushion_mm']}M\n"
                            f"  Z {v['credit_score']['z_score']} "
                            f"({v['credit_score']['zone']}) | {rec}\n"
                            f"  Lev {r['net_leverage']}x (lim {r['leverage_limit']}x) | "
                            f"Breach {r['covenant_breached']} | "
                            f"Cov {r['interest_coverage']}x | "
                            f"LTV {r['ltv_pct']}% | Sent {r['sentiment_score']}"
                        )
                    context = "\n".join(lines)
                else:
                    r = df[df["borrower"]==borrower].iloc[0]
                    v = full_valuation(r,"direct_lending")
                    rt = v["rate_components"]
                    rec = (
                        f"Recovery ${v['recovery']['recovery_value_mm']}M "
                        f"({round(v['recovery']['recovery_rate']*100,1)}%), "
                        f"LGD {round(v['recovery']['loss_given_default']*100,1)}%, "
                        f"asset cov {v['recovery']['asset_coverage_ratio']}x"
                        if v.get("recovery") else "No covenant breach"
                    )
                    context = f"""
BORROWER: {borrower} ({r['sector']})
NAV: ${v['blended_nav_mm']}M ({v['nav_to_par_pct']}% par) — {v['nav_flag']}
Weighting: {v['weighting']} | {v['asc820_level']}
Rate: {rt['total_rate_pct']}% [SOFR {rt['sofr_base']}% + spread {rt['credit_spread']}% {rt['credit_bucket']} + illiq {rt['illiquidity_prem']}% + cxp {rt['complexity_prem']}% + sec {rt['sector_adj']}% + cov {rt['covenant_adj']}%] BSL prem {rt['bsl_premium_pct']}%
DCF: ${v['dcf']['nav']}M ({v['dcf']['nav_to_par']}% par) | Dur {v['dcf']['duration_yrs']}yr | DV01 ${v['dcf']['dv01_mm']}M/100bps
MC: P10 ${v['monte_carlo']['p10']}M | P25 ${v['monte_carlo']['p25']}M | P50 ${v['monte_carlo']['p50']}M | P75 ${v['monte_carlo']['p75']}M | P90 ${v['monte_carlo']['p90']}M | PD {v['monte_carlo']['pd_applied']}% | E[Loss] ${v['monte_carlo']['expected_loss_mm']}M
Comps: EV ${v['comps']['implied_ev']}M | Cov {v['comps']['ev_coverage']}x | Cushion ${v['comps']['equity_cushion_mm']}M ({v['comps']['equity_cushion_pct']}%) | Bear ${v['comps']['ev_bear_case']}M | Bull ${v['comps']['ev_bull_case']}M | {v['comps']['coverage_assessment']}
Z: {v['credit_score']['z_score']} — {v['credit_score']['zone']}
Recovery: {rec}
Signals: Lev {r['net_leverage']}x (lim {r['leverage_limit']}x) | Breach {r['covenant_breached']} | Cov {r['interest_coverage']}x | LTV {r['ltv_pct']}% | Sent {r['sentiment_score']} | Flags {int(r['risk_flags'])}/4
News: {r['recent_news']}"""

                prompt = f"""You are a senior private credit analyst presenting to a CIO.

STRICT RULES:
1. Every sentence must contain at least one specific number or percentage.
2. Never make a general statement without backing it with data.
3. Be direct and decisive. No hedging. No preamble.
4. 5 to 7 sentences only. Be concise but complete.
5. Start your answer immediately — no "Based on the data" opener.

Portfolio data:
{context}

Question: {question}

Answer:"""

                resp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=500,
                    messages=[{"role":"user","content":prompt}]
                )
                answer = resp.choices[0].message.content

            st.markdown("---")
            st.markdown(
                f'<div style="background:#EFF6FF;border-left:4px solid #3B82F6;'
                f'border-radius:4px;padding:10px 14px;margin-bottom:8px;'
                f'font-size:0.85rem;font-weight:600;color:#1E40AF;">Q: {question}</div>',
                unsafe_allow_html=True
            )
            st.info(answer)
            st.session_state["ask_q"] = ""