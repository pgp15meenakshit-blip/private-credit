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

st.set_page_config(
    page_title="Private Credit Intelligence",
    page_icon="📊",
    layout="wide"
)

# ── GLOBAL STYLES ─────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #888; }
.section-header {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    color: #888; text-transform: uppercase; margin-bottom: 0.4rem;
}
.memo-box {
    border-radius: 8px; padding: 16px 20px;
    margin-bottom: 12px; line-height: 1.8;
    font-size: 0.9rem;
}
.status-healthy { background:#F0FFF4; border-left:4px solid #38A169; }
.status-watch   { background:#FFFBEB; border-left:4px solid #D69E2E; }
.status-risk    { background:#FFF5F5; border-left:4px solid #E53E3E; }
</style>
""", unsafe_allow_html=True)

# ── DATA & SHARED FUNCTIONS ───────────────────────────────────────
df = load_portfolio()

RADAR_CATS = ["Leverage", "Coverage", "LTV",
              "Sentiment", "Risk Flags", "Yield"]

def radar_vals(row):
    return [
        min(row["net_leverage"] / 6.0, 1.0),
        max(0, 1 - (row["interest_coverage"] - 1) / 4.0),
        row["ltv_pct"] / 100,
        max(0, (-row["sentiment_score"] + 1) / 2),
        row["risk_flags"] / 4.0,
        min(row["coupon_pct"] / 15.0, 1.0),
    ]

def status_badge(flags):
    if flags >= 3:
        return "🔴 HIGH RISK", "status-risk"
    elif flags >= 1:
        return "🟡 WATCH", "status-watch"
    return "🟢 HEALTHY", "status-healthy"

def render_memo(memo_text):
    """Render AI memo with colour-coded sections."""
    section_styles = {
        "HEALTH SUMMARY":       ("#EBF8FF", "#2B6CB0"),
        "VALUATION ASSESSMENT": ("#F0FFF4", "#276749"),
        "KEY RISKS":            ("#FFFBEB", "#744210"),
        "RECOMMENDED ACTION":   ("#FAF5FF", "#553C9A"),
    }
    current_section = None
    current_lines   = []

    def flush(sec, lines):
        if not sec or not lines:
            return
        bg, accent = section_styles.get(sec, ("#F7FAFC", "#2D3748"))
        content = " ".join(lines).replace(
            "**", "").replace("##", "")
        st.markdown(
            f'<div class="memo-box" style="background:{bg};'
            f'border-left:4px solid {accent};">'
            f'<div style="font-size:0.7rem;font-weight:700;'
            f'letter-spacing:0.08em;color:{accent};'
            f'text-transform:uppercase;margin-bottom:8px;">{sec}</div>'
            f'<div style="color:#2D3748;">{content}</div></div>',
            unsafe_allow_html=True
        )

    for line in memo_text.split("\n"):
        matched = False
        for sec in section_styles:
            if sec in line.upper():
                flush(current_section, current_lines)
                current_section = sec
                current_lines   = []
                matched = True
                break
        if not matched and line.strip():
            current_lines.append(line.strip())
    flush(current_section, current_lines)


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Private Credit Intelligence")
    st.caption("AI-powered · ASC 820 compliant")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Overview",
        "⚠️  Risk Watchlist",
        "📐  Valuation Engine",
        "🤖  Credit Memo",
        "📈  NAV Scenarios",
        "💬  Ask the AI",
    ])
    st.markdown("---")
    st.markdown('<div class="section-header">Methodology</div>',
                unsafe_allow_html=True)
    st.caption("Discount rate: Build-up method")
    st.caption("DCF 50% · Comps 30% · MC P50 20%")
    st.caption("Framework: ASC 820 / IFRS 13")
    st.caption("Monte Carlo: 10,000 simulations")
    st.caption("Credit score: Adapted Altman Z")


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Portfolio Overview")
    st.caption(f"Monitoring {len(df)} loans · "
               f"${df['principal_mm'].sum():.0f}M total exposure")

    # ── KPI ROW ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Loans",       len(df))
    c2.metric("Total Exposure",    f"${df['principal_mm'].sum():.0f}M")
    c3.metric("Loans at Risk",     int((df['risk_flags'] > 0).sum()))
    c4.metric("Avg Coupon",        f"{df['coupon_pct'].mean():.1f}%")
    c5.metric("Covenant Breaches", int(df['covenant_breached'].sum()))

    st.markdown("---")

    # ── RISK HEATMAP ─────────────────────────────────────────────
    st.subheader("Portfolio Risk Heatmap")
    st.caption("Each cell scored 0–1 · Green = healthy · Red = stressed")

    heat_rows = []
    for _, row in df.iterrows():
        heat_rows.append({
            "Borrower":   row["borrower"],
            "Leverage":   round(min(row["net_leverage"] / 6.0, 1.0), 2),
            "Coverage":   round(max(0, 1-(row["interest_coverage"]-1)/4), 2),
            "LTV":        round(row["ltv_pct"] / 100, 2),
            "Sentiment":  round(max(0, (-row["sentiment_score"]+1)/2), 2),
            "Risk Flags": round(row["risk_flags"] / 4.0, 2),
            "Yield":      round(min(row["coupon_pct"] / 15.0, 1.0), 2),
        })
    heat_df = pd.DataFrame(heat_rows).set_index("Borrower")

    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=["#48BB78", "#ECC94B", "#FC8181"],
        zmin=0, zmax=1, text_auto=".2f", aspect="auto",
        title="Risk Score Matrix — All Borrowers"
    )
    fig_heat.update_layout(
        coloraxis_colorbar=dict(title="Risk Score"),
        height=380
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── CHARTS ROW ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Exposure by Borrower")
        fig_exp = px.bar(
            df, x="borrower", y="principal_mm", color="sector",
            labels={"principal_mm": "Principal ($M)",
                    "borrower": ""},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_exp.update_layout(
            showlegend=True, height=300,
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    with col2:
        st.subheader("Leverage vs Coverage")
        fig_lc = px.scatter(
            df, x="net_leverage", y="interest_coverage",
            color="sector", size="principal_mm",
            hover_name="borrower",
            labels={"net_leverage": "Net Leverage (x)",
                    "interest_coverage": "Interest Coverage (x)"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_lc.add_hline(y=2.0, line_dash="dash",
                         line_color="#E53E3E",
                         annotation_text="Coverage floor 2x")
        fig_lc.add_vline(x=5.0, line_dash="dash",
                         line_color="#E53E3E",
                         annotation_text="Leverage limit 5x")
        fig_lc.update_layout(height=300)
        st.plotly_chart(fig_lc, use_container_width=True)

    st.markdown("---")

    # ── RADAR COMPARISON ─────────────────────────────────────────
    st.subheader("Borrower Risk Profile Comparison")
    col1, col2 = st.columns(2)
    b1 = col1.selectbox("Borrower A", df["borrower"].tolist(),
                        index=1, key="ra1")
    b2 = col2.selectbox("Borrower B", df["borrower"].tolist(),
                        index=2, key="ra2")

    fig_radar = go.Figure()
    for bname, colour in [(b1, "#4299E1"), (b2, "#ED8936")]:
        row  = df[df["borrower"] == bname].iloc[0]
        vals = radar_vals(row) + [radar_vals(row)[0]]
        cats = RADAR_CATS + [RADAR_CATS[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=bname, line_color=colour, opacity=0.75
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        title="Risk Profile — higher score = more risk",
        height=400, showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — RISK WATCHLIST
# ══════════════════════════════════════════════════════════════════
elif "Watchlist" in page:
    st.title("Risk Watchlist")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flagged Loans",     int((df['risk_flags'] > 0).sum()))
    c2.metric("Covenant Breaches", int(df['covenant_breached'].sum()))
    c3.metric("Low Coverage",      int(df['low_coverage'].sum()))
    c4.metric("High LTV (>75%)",   int(df['high_ltv'].sum()))

    st.markdown("---")

    watchlist = df[df["risk_flags"] > 0].sort_values(
        "risk_flags", ascending=False)

    if watchlist.empty:
        st.success("No loans currently flagged.")
    else:
        # Status cards
        for _, row in watchlist.iterrows():
            badge, style = status_badge(int(row["risk_flags"]))
            with st.expander(
                f"{badge}  {row['borrower']} — "
                f"{row['sector']} · "
                f"${row['principal_mm']}M · "
                f"{int(row['risk_flags'])}/4 flags"
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Net Leverage",
                            f"{row['net_leverage']}x",
                            delta=f"limit {row['leverage_limit']}x",
                            delta_color="inverse")
                col2.metric("Coverage",
                            f"{row['interest_coverage']}x")
                col3.metric("LTV",
                            f"{row['ltv_pct']}%")
                col4.metric("Sentiment",
                            f"{row['sentiment_score']}")
                st.caption(f"News: {row['recent_news']}")
                flags_triggered = []
                if row["covenant_breached"]:
                    flags_triggered.append("Covenant breached")
                if row["low_coverage"]:
                    flags_triggered.append("Low interest coverage")
                if row["high_ltv"]:
                    flags_triggered.append("High LTV")
                if row["neg_sentiment"]:
                    flags_triggered.append("Negative sentiment")
                st.error("Flags: " + " · ".join(flags_triggered))

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            fig_flags = px.bar(
                watchlist, x="borrower", y="risk_flags",
                color="risk_flags",
                color_continuous_scale=["#ECC94B","#FC8181","#C53030"],
                title="Risk Flag Count by Borrower",
                labels={"risk_flags": "Flags", "borrower": ""}
            )
            fig_flags.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_flags, use_container_width=True)

        with col2:
            fig_r = go.Figure()
            colours = ["#FC8181","#F6AD55","#ECC94B","#68D391","#63B3ED"]
            for i, (_, row) in enumerate(watchlist.iterrows()):
                vals = radar_vals(row) + [radar_vals(row)[0]]
                cats = RADAR_CATS + [RADAR_CATS[0]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself",
                    name=row["borrower"],
                    line_color=colours[i % len(colours)],
                    opacity=0.65
                ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(range=[0,1])),
                title="Stressed Borrower Profiles",
                height=300
            )
            st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — VALUATION ENGINE
# ══════════════════════════════════════════════════════════════════
elif "Valuation" in page:
    st.title("Valuation Engine")
    st.caption("ASC 820 / IFRS 13 · Build-up discount rate · "
               "DCF · EV/EBITDA comps · Monte Carlo")

    col1, col2 = st.columns([2, 1])
    borrower   = col1.selectbox("Borrower",
                                df["borrower"].tolist(), key="ve_b")
    instrument = col2.selectbox("Instrument type", [
        "direct_lending", "unitranche", "second_lien",
        "mezzanine", "pik", "asset_based"
    ])
    row = df[df["borrower"] == borrower].iloc[0]

    # ── INTERACTIVE DISCOUNT RATE BUILDER ────────────────────────
    st.markdown("---")
    st.subheader("Interactive Discount Rate Builder")
    st.caption("Adjust any component — DCF NAV updates in real time")

    base = build_discount_rate(row, instrument)

    sc1, sc2, sc3 = st.columns(3)
    sofr = sc1.slider("SOFR base (%)",       2.0,  8.0,
                      float(base["sofr_base"]),        0.05)
    csp  = sc2.slider("Credit spread (%)",   0.5, 10.0,
                      float(base["credit_spread"]),    0.05)
    ill  = sc3.slider("Illiquidity prem (%)",0.5,  4.0,
                      float(base["illiquidity_prem"]), 0.05)

    sc4, sc5, sc6 = st.columns(3)
    cxp  = sc4.slider("Complexity prem (%)", 0.0,  2.0,
                      float(base["complexity_prem"]),  0.05)
    sec  = sc5.slider("Sector adj (%)",     -0.5,  1.5,
                      float(base["sector_adj"]),       0.05)
    cov  = sc6.slider("Covenant adj (%)",    0.0,  2.0,
                      float(base["covenant_adj"]),     0.05)

    live_rate = sofr + csp + ill + cxp + sec + cov
    live_dcf  = dcf_valuation(row, live_rate)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Discount Rate",   f"{live_rate:.2f}%")
    m2.metric("DCF NAV",         f"${live_dcf['nav']:.2f}M",
              delta=f"{live_dcf['nav_to_par']:.1f}% of par")
    m3.metric("Duration",        f"{live_dcf['duration_yrs']:.2f}yr")
    m4.metric("DV01",            f"${live_dcf['dv01_mm']:.3f}M / 100bps")

    names = ["SOFR","Credit spread","Illiquidity",
             "Complexity","Sector adj","Covenant adj"]
    vals  = [sofr, csp, ill, cxp, sec, cov]
    fig_wf = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=["#4299E1","#ECC94B","#48BB78",
                      "#9F7AEA","#FC8181","#E53E3E"],
        text=[f"{v:.2f}%" for v in vals],
        textposition="outside"
    ))
    fig_wf.add_hline(
        y=live_rate, line_dash="dash", line_color="#E53E3E",
        annotation_text=f"Total: {live_rate:.2f}%"
    )
    fig_wf.update_layout(
        title="Discount Rate Build-Up — drag sliders to update",
        yaxis_title="Rate (%)", showlegend=False, height=320
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("---")

    # ── FULL VALUATION ────────────────────────────────────────────
    if st.button("▶  Run Full Valuation", type="primary"):
        with st.spinner("Running DCF · Comps · Monte Carlo · Z-score..."):
            val = full_valuation(row, instrument)
        st.session_state[f"val_{borrower}"] = val

    if f"val_{borrower}" in st.session_state:
        val = st.session_state[f"val_{borrower}"]
        r   = val["rate_components"]
        d   = val["dcf"]
        c   = val["comps"]
        m   = val["monte_carlo"]
        z   = val["credit_score"]

        badge_colour = (
            "#FC8181" if val["nav_to_par_pct"] < 90
            else "#ECC94B" if val["nav_to_par_pct"] < 98
            else "#48BB78"
        )
        st.markdown(
            f'<div style="background:{badge_colour}22;border-left:'
            f'4px solid {badge_colour};border-radius:4px;'
            f'padding:10px 16px;margin-bottom:1rem;">'
            f'<b>{val["nav_flag"]}</b> · Blended NAV '
            f'<b>${val["blended_nav_mm"]}M</b> '
            f'({val["nav_to_par_pct"]}% of par) · '
            f'{val["asc820_level"]} · {val["weighting"]}</div>',
            unsafe_allow_html=True
        )

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Blended NAV",   f"${val['blended_nav_mm']}M",
                  f"{val['nav_to_par_pct']}% of par")
        h2.metric("Discount Rate", f"{r['total_rate_pct']}%",
                  f"BSL premium: {r['bsl_premium_pct']}%")
        h3.metric("Z-Score",       f"{z['z_score']}",
                  z['zone'])
        h4.metric("Default Prob",  f"{m['pd_applied']}%",
                  f"Exp. loss ${m['expected_loss_mm']}M")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("DCF Cash Flow Analysis")
            cfs = d["cash_flows"]
            fig_dcf = go.Figure()
            fig_dcf.add_bar(
                x=[f"Year {c['period']}" for c in cfs],
                y=[c["cash_flow"]         for c in cfs],
                name="Cash Flow ($M)", marker_color="#4299E1"
            )
            fig_dcf.add_bar(
                x=[f"Year {c['period']}" for c in cfs],
                y=[c["present_value"]    for c in cfs],
                name="Present Value ($M)", marker_color="#ECC94B"
            )
            fig_dcf.update_layout(
                barmode="group", height=300,
                title=f"DCF NAV ${d['nav']}M · {d['nav_to_par']}% of par"
            )
            st.plotly_chart(fig_dcf, use_container_width=True)
            st.caption(
                f"Duration: {d['duration_yrs']}yr · "
                f"DV01: ${d['dv01_mm']}M per 100bps · "
                f"Discount rate: {d['discount_rate']}%"
            )

        with col2:
            st.subheader("Monte Carlo Distribution")
            stress = st.toggle("Stress scenario — double default probability",
                               key="mc_stress")
            if stress:
                with st.spinner("Running stress scenario..."):
                    row_s = row.copy()
                    row_s["risk_flags"] = min(int(row["risk_flags"])+2, 4)
                    mc_s = monte_carlo_valuation(row_s, instrument)
                navs  = np.array(mc_s["nav_distribution"])
                p10, p50, p90 = mc_s["p10"], mc_s["p50"], mc_s["p90"]
                pd_lbl = f"Stress PD: {mc_s['pd_applied']}%"
                bar_col = "#FC8181"
            else:
                navs  = np.array(m["nav_distribution"])
                p10, p50, p90 = m["p10"], m["p50"], m["p90"]
                pd_lbl = f"Base PD: {m['pd_applied']}%"
                bar_col = "#48BB78"

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=navs, nbinsx=60,
                marker_color=bar_col, opacity=0.75,
                name="Simulated NAVs"
            ))
            for lbl, vp, col in [
                ("P10", p10, "#E53E3E"),
                ("P50", p50, "#D69E2E"),
                ("P90", p90, "#38A169")
            ]:
                fig_mc.add_vline(
                    x=vp, line_dash="dash", line_color=col,
                    annotation_text=f"{lbl} ${vp}M"
                )
            fig_mc.update_layout(
                title=f"Monte Carlo 10,000 sims · {pd_lbl}",
                xaxis_title="NAV ($M)",
                yaxis_title="Frequency", height=300
            )
            st.plotly_chart(fig_mc, use_container_width=True)
            st.caption(
                f"P10: ${p10}M · P50: ${p50}M · P90: ${p90}M · "
                f"Expected loss: ${m['expected_loss_mm']}M"
            )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Comparable Company Analysis")
            comps_rows = [
                ["Sector Multiple",  f"{c['sector_multiple']}x EV/EBITDA"],
                ["Implied EV",       f"${c['implied_ev']}M"],
                ["Debt Coverage",    f"{c['ev_coverage']}x"],
                ["Equity Cushion",   f"${c['equity_cushion_mm']}M "
                                     f"({c['equity_cushion_pct']}%)"],
                ["Bear Case EV",     f"${c['ev_bear_case']}M"],
                ["Bull Case EV",     f"${c['ev_bull_case']}M"],
            ]
            st.dataframe(
                pd.DataFrame(comps_rows, columns=["Metric","Value"]),
                hide_index=True, use_container_width=True
            )
            badge, style = status_badge(
                0 if c["ev_coverage"] > 2.0
                else 1 if c["ev_coverage"] > 1.3 else 3
            )
            st.caption(c["coverage_assessment"])

        with col2:
            st.subheader("Altman Z-Score Breakdown")
            zc  = z["components"]
            fig_z = go.Figure(go.Bar(
                x=list(zc.keys()),
                y=list(zc.values()),
                marker_color=["#4299E1","#48BB78","#ECC94B",
                              "#9F7AEA","#FC8181"],
                text=[f"{v:.3f}" for v in zc.values()],
                textposition="outside"
            ))
            fig_z.update_layout(
                title=f"Z = {z['z_score']} — {z['zone']}",
                yaxis_title="Component Score",
                height=300
            )
            st.plotly_chart(fig_z, use_container_width=True)

        if val.get("recovery"):
            st.markdown("---")
            st.error("⚠️ Covenant Breach — Recovery Analysis Active")
            rec = val["recovery"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Recovery Rate",
                      f"{rec['recovery_rate']*100:.1f}%")
            r2.metric("Recovery Value",
                      f"${rec['recovery_value_mm']}M")
            r3.metric("Loss Given Default",
                      f"{rec['loss_given_default']*100:.1f}%")
            r4.metric("Asset Coverage",
                      f"{rec['asset_coverage_ratio']}x")


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — CREDIT MEMO GENERATOR
# ══════════════════════════════════════════════════════════════════
elif "Memo" in page:
    st.title("AI Credit Memo Generator")
    st.caption("Institutional-grade credit narratives grounded "
               "in full valuation stack")

    col1, col2 = st.columns([2, 1])
    borrower   = col1.selectbox("Borrower",
                                df["borrower"].tolist(), key="cm_b")
    instrument = col2.selectbox("Instrument type", [
        "direct_lending","unitranche","second_lien",
        "mezzanine","pik","asset_based"
    ], key="cm_i")
    row   = df[df["borrower"] == borrower].iloc[0]
    flags = int(row["risk_flags"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Leverage",
              f"{row['net_leverage']}x",
              delta=f"limit {row['leverage_limit']}x",
              delta_color="inverse")
    c2.metric("Coverage", f"{row['interest_coverage']}x")
    c3.metric("LTV",      f"{row['ltv_pct']}%")
    c4.metric("Sentiment",f"{row['sentiment_score']}")

    badge, style = status_badge(flags)
    st.markdown(
        f'<div class="memo-box {style}">'
        f'<b>{badge}</b> — {flags} of 4 risk flags triggered · '
        f'{row["recent_news"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("⚡  Generate Credit Memo", type="primary"):
        with st.spinner("Running valuation engine and generating "
                        "institutional memo..."):
            val  = full_valuation(row, instrument)
            memo = generate_credit_narrative(row, val)
        st.session_state[f"memo_{borrower}"]    = memo
        st.session_state[f"memoval_{borrower}"] = val

    if f"memo_{borrower}" in st.session_state:
        memo = st.session_state[f"memo_{borrower}"]
        val  = st.session_state[f"memoval_{borrower}"]

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Blended NAV",
                  f"${val['blended_nav_mm']}M",
                  f"{val['nav_to_par_pct']}% of par")
        v2.metric("Discount Rate",
                  f"{val['rate_components']['total_rate_pct']}%")
        v3.metric("Z-Score",
                  f"{val['credit_score']['z_score']}")
        v4.metric("Default Prob",
                  f"{val['monte_carlo']['pd_applied']}%")

        st.markdown("---")
        st.subheader(f"Credit Memo — {borrower}")
        st.caption(
            f"Generated {val['valuation_date']} · "
            f"{val['asc820_level']} · {val['methodology_note']}"
        )
        render_memo(memo)
        st.markdown("---")
        st.download_button(
            "📥  Download memo (.txt)", memo,
            file_name=f"{borrower}_credit_memo.txt"
        )


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — NAV SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif "NAV" in page:
    st.title("NAV Scenario Analysis")
    st.caption("Real-time portfolio valuation sensitivity")

    rate = st.slider(
        "Discount Rate (%)", 4.0, 20.0, 10.0,
        step=0.25, format="%.2f%%"
    )

    def loan_nav(row, dr):
        d  = dr / 100
        p  = row["principal_mm"]
        c  = p * row["coupon_pct"] / 100
        cf = [c] * 4; cf[-1] += p
        return sum(v/(1+d)**t for t, v in enumerate(cf, 1))

    df["s_nav"] = df.apply(lambda r: loan_nav(r, rate), axis=1)
    tot_nav = df["s_nav"].sum()
    tot_par = df["principal_mm"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio NAV",  f"${tot_nav:.1f}M")
    c2.metric("Par Value",      f"${tot_par:.0f}M")
    c3.metric("NAV vs Par",     f"${tot_nav-tot_par:.1f}M",
              delta_color="normal")
    c4.metric("Avg NAV / Par",  f"{tot_nav/tot_par*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(
            df, x="borrower", y="s_nav", color="sector",
            title=f"Loan NAV at {rate:.2f}% Discount Rate ($M)",
            labels={"s_nav": "NAV ($M)", "borrower": ""},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_layout(height=320, xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        rates = np.arange(4.0, 20.25, 0.25)
        navs  = [df.apply(lambda r: loan_nav(r, dr),
                          axis=1).sum() for dr in rates]
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=rates, y=navs, mode="lines",
            line=dict(color="#4299E1", width=2),
            fill="tozeroy",
            fillcolor="rgba(66,153,225,0.1)",
            name="Portfolio NAV"
        ))
        fig_curve.add_vline(
            x=rate, line_dash="dash", line_color="#E53E3E",
            annotation_text=f"Current: {rate:.2f}%"
        )
        fig_curve.add_hline(
            y=tot_par, line_dash="dot", line_color="#718096",
            annotation_text="Par value"
        )
        fig_curve.update_layout(
            title="NAV Sensitivity Curve",
            xaxis_title="Discount Rate (%)",
            yaxis_title="Portfolio NAV ($M)",
            height=320
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    # ── DV01 TABLE ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("DV01 — Rate Sensitivity per Loan")
    st.caption("$ change in NAV for a 100bps increase in discount rate")

    dv_rows = []
    for _, row in df.iterrows():
        n_base = loan_nav(row, rate)
        n_up   = loan_nav(row, rate + 1.0)
        dv_rows.append({
            "Borrower":       row["borrower"],
            "Sector":         row["sector"],
            "DV01 ($M)":      round(n_base - n_up, 3),
            "Current NAV":    f"${n_base:.2f}M",
            "NAV +100bps":    f"${n_up:.2f}M",
        })
    dv_df = pd.DataFrame(dv_rows).sort_values("DV01 ($M)",
                                               ascending=False)
    st.dataframe(dv_df, hide_index=True, use_container_width=True)

    fig_dv = px.bar(
        dv_df, x="Borrower", y="DV01 ($M)", color="Sector",
        title="DV01 by Borrower — most rate-sensitive loans first",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_dv.update_layout(height=300, xaxis_tickangle=-30)
    st.plotly_chart(fig_dv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 6 — ASK THE AI
# ══════════════════════════════════════════════════════════════════
elif "Ask" in page:
    st.title("Ask the AI")
    st.caption("Free-text Q&A grounded in live valuation data "
               "— the agentic layer")

    borrower = st.selectbox(
        "Borrower context",
        ["All borrowers"] + df["borrower"].tolist(),
        key="ai_b"
    )

    if "ask_q" not in st.session_state:
        st.session_state["ask_q"] = ""

    st.markdown("**Suggested questions:**")
    col1, col2, col3 = st.columns(3)

    if col1.button("Why is BetaCorp below par?"):
        st.session_state["ask_q"] = (
            "Why is BetaCorp trading at only 89.4% of par? "
            "Break down every contributing factor — discount rate "
            "components, covenant breach, Z-score, Monte Carlo "
            "default probability, and recovery risk. "
            "Reference specific numbers throughout."
        )
    if col2.button("SOFR drops 100bps — impact?"):
        st.session_state["ask_q"] = (
            "If SOFR drops by exactly 100 basis points from 4.33% "
            "to 3.33%, calculate the new NAV for each borrower "
            "using their DV01 and show the total portfolio impact "
            "in dollars. Rank borrowers by sensitivity."
        )
    if col3.button("Worst recovery profile?"):
        st.session_state["ask_q"] = (
            "Which borrower has the worst recovery profile? "
            "Compare recovery rates, LGD, asset coverage ratio, "
            "and default probability across all borrowers. "
            "Rank them from worst to best and explain why."
        )

    question = st.text_input(
        "Your question",
        value=st.session_state["ask_q"],
        placeholder="e.g. Which loans are most exposed to a rate rise?"
    )
    st.session_state["ask_q"] = question

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analysing portfolio data..."):

                if borrower == "All borrowers":
                    lines = []
                    for _, r in df.iterrows():
                        v = full_valuation(r, "direct_lending")
                        rec = (
                            f"Recovery ${v['recovery']['recovery_value_mm']}M "
                            f"({round(v['recovery']['recovery_rate']*100,1)}%), "
                            f"LGD {round(v['recovery']['loss_given_default']*100,1)}%"
                            if v.get("recovery") else "No breach"
                        )
                        lines.append(
                            f"\n{r['borrower']} ({r['sector']}):\n"
                            f"  NAV ${v['blended_nav_mm']}M "
                            f"({v['nav_to_par_pct']}% par) — "
                            f"{v['nav_flag']}\n"
                            f"  Rate: {v['rate_components']['total_rate_pct']}% "
                            f"[SOFR {v['rate_components']['sofr_base']}% + "
                            f"spread {v['rate_components']['credit_spread']}% + "
                            f"illiq {v['rate_components']['illiquidity_prem']}% + "
                            f"cov adj {v['rate_components']['covenant_adj']}%]\n"
                            f"  DCF ${v['dcf']['nav']}M | "
                            f"DV01 ${v['dcf']['dv01_mm']}M | "
                            f"Duration {v['dcf']['duration_yrs']}yr\n"
                            f"  MC P10 ${v['monte_carlo']['p10']}M | "
                            f"P50 ${v['monte_carlo']['p50']}M | "
                            f"P90 ${v['monte_carlo']['p90']}M | "
                            f"PD {v['monte_carlo']['pd_applied']}% | "
                            f"E[Loss] ${v['monte_carlo']['expected_loss_mm']}M\n"
                            f"  Comps EV ${v['comps']['implied_ev']}M | "
                            f"Coverage {v['comps']['ev_coverage']}x | "
                            f"Cushion ${v['comps']['equity_cushion_mm']}M\n"
                            f"  Z-score {v['credit_score']['z_score']} "
                            f"({v['credit_score']['zone']})\n"
                            f"  {rec}\n"
                            f"  Leverage {r['net_leverage']}x "
                            f"(limit {r['leverage_limit']}x) | "
                            f"Breach {r['covenant_breached']} | "
                            f"Coverage {r['interest_coverage']}x | "
                            f"LTV {r['ltv_pct']}% | "
                            f"Sentiment {r['sentiment_score']}"
                        )
                    context = "\n".join(lines)

                else:
                    r = df[df["borrower"] == borrower].iloc[0]
                    v = full_valuation(r, "direct_lending")
                    rt = v["rate_components"]
                    rec = (
                        f"Recovery ${v['recovery']['recovery_value_mm']}M "
                        f"({round(v['recovery']['recovery_rate']*100,1)}%), "
                        f"LGD {round(v['recovery']['loss_given_default']*100,1)}%, "
                        f"asset coverage {v['recovery']['asset_coverage_ratio']}x"
                        if v.get("recovery") else "No covenant breach"
                    )
                    context = f"""
BORROWER: {borrower} ({r['sector']})
Blended NAV: ${v['blended_nav_mm']}M ({v['nav_to_par_pct']}% of par) — {v['nav_flag']}
Weighting: {v['weighting']} | {v['asc820_level']}

DISCOUNT RATE: {rt['total_rate_pct']}% total
  SOFR {rt['sofr_base']}% + Credit spread {rt['credit_spread']}% [{rt['credit_bucket']}]
  + Illiquidity {rt['illiquidity_prem']}% + Complexity {rt['complexity_prem']}%
  + Sector adj {rt['sector_adj']}% + Covenant adj {rt['covenant_adj']}%
  BSL premium: {rt['bsl_premium_pct']}%

DCF: ${v['dcf']['nav']}M ({v['dcf']['nav_to_par']}% par) | Duration {v['dcf']['duration_yrs']}yr | DV01 ${v['dcf']['dv01_mm']}M/100bps
MONTE CARLO: P10 ${v['monte_carlo']['p10']}M | P25 ${v['monte_carlo']['p25']}M | P50 ${v['monte_carlo']['p50']}M | P75 ${v['monte_carlo']['p75']}M | P90 ${v['monte_carlo']['p90']}M
  PD: {v['monte_carlo']['pd_applied']}% | Expected loss: ${v['monte_carlo']['expected_loss_mm']}M
COMPS: EV ${v['comps']['implied_ev']}M | Coverage {v['comps']['ev_coverage']}x | Cushion ${v['comps']['equity_cushion_mm']}M ({v['comps']['equity_cushion_pct']}%)
  Bear ${v['comps']['ev_bear_case']}M | Bull ${v['comps']['ev_bull_case']}M | {v['comps']['coverage_assessment']}
CREDIT SCORE: Z = {v['credit_score']['z_score']} — {v['credit_score']['zone']}
RECOVERY: {rec}
SIGNALS: Leverage {r['net_leverage']}x (limit {r['leverage_limit']}x) | Breach {r['covenant_breached']}
  Coverage {r['interest_coverage']}x | LTV {r['ltv_pct']}% | Sentiment {r['sentiment_score']}
  Flags: {int(r['risk_flags'])}/4 | News: {r['recent_news']}
"""

                prompt = f"""You are a senior private credit analyst \
presenting to a CIO.

RULES — follow strictly:
1. Every sentence must contain at least one specific number.
2. Never make a general statement — always back with data.
3. Be direct and decisive — no hedging, no preamble.
4. Structure clearly. Use numbers, percentages, dollar amounts.
5. 5–7 sentences maximum.
6. Start your answer directly — no "Based on the data" opener.

Portfolio data:
{context}

Question: {question}

Answer:"""

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=500,
                    messages=[{"role":"user","content":prompt}]
                )
                answer = response.choices[0].message.content

            st.markdown("---")
            st.markdown(f"**Q: {question}**")
            st.info(answer)
            st.session_state["ask_q"] = ""