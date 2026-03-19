import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data import load_portfolio
from valuation import full_valuation, build_discount_rate, dcf_valuation
from narrative import generate_credit_narrative
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="IVP Credit Intelligence",
    page_icon="📊",
    layout="wide"
)

# ── HELPERS ───────────────────────────────────────────────────────
def risk_colour(val, low, high):
    if val <= low:  return "#6BC99A"
    if val <= high: return "#FFD166"
    return "#FF6B6B"

def section_box(title, content, colour="#E6F1FB", text_colour="#0C447C"):
    st.markdown(
        f"""<div style="background:{colour};border-radius:8px;
        padding:14px 18px;margin-bottom:10px;">
        <div style="font-size:11px;font-weight:600;letter-spacing:0.07em;
        color:{text_colour};text-transform:uppercase;margin-bottom:6px;">
        {title}</div>
        <div style="font-size:13px;color:#1a1a2e;line-height:1.75;">
        {content}</div></div>""",
        unsafe_allow_html=True
    )

df = load_portfolio()
def radar_values(row):
    return [
        min(row["net_leverage"] / 6.0, 1.0),
        max(0, 1 - (row["interest_coverage"] - 1) / 4.0),
        row["ltv_pct"] / 100,
        max(0, (-row["sentiment_score"] + 1) / 2),
        row["risk_flags"] / 4.0,
        min(row["coupon_pct"] / 15.0, 1.0),
    ]

cats = ["Leverage Risk", "Coverage Risk", "LTV Risk",
        "Sentiment Risk", "Flag Score", "Yield Level"]

# ── SIDEBAR ───────────────────────────────────────────────────────
st.sidebar.title("IVP Credit Intelligence")
st.sidebar.caption("Private Credit · AI-Powered")
page = st.sidebar.radio("", [
    " Portfolio Overview",
    " Risk Watchlist",
    " Valuation Engine",
    " Credit Memo Generator",
    " NAV Scenario Analysis",
    " Ask the AI",
])
st.sidebar.markdown("---")
st.sidebar.caption("Framework: ASC 820 / IFRS 13")
st.sidebar.caption("Discount rate: Build-up method")
st.sidebar.caption("Weighting: DCF 50% · Comps 30% · MC 20%")
st.sidebar.caption("Monte Carlo: 10,000 simulations")


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Portfolio Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Loans",       len(df))
    c2.metric("Total Exposure",    f"${df['principal_mm'].sum():.0f}M")
    c3.metric("Loans at Risk",     int((df['risk_flags'] > 0).sum()))
    c4.metric("Avg Coupon",        f"{df['coupon_pct'].mean():.1f}%")
    c5.metric("Covenant Breaches", int(df['covenant_breached'].sum()))

    st.markdown("---")

    # ── RISK HEATMAP ─────────────────────────────────────────────
    st.subheader("Portfolio Risk Heatmap")
    st.caption("Green = healthy · Amber = watch · Red = stressed")

    heat_data = []
    for _, row in df.iterrows():
        lev_score  = min(row["net_leverage"] / 6.0, 1.0)
        cov_score  = max(0, 1 - (row["interest_coverage"] - 1) / 4.0)
        ltv_score  = row["ltv_pct"] / 100
        sent_score = max(0, (-row["sentiment_score"] + 1) / 2)
        flag_score = row["risk_flags"] / 4.0
        heat_data.append({
            "Borrower":   row["borrower"],
            "Leverage":   round(lev_score, 2),
            "Coverage":   round(cov_score, 2),
            "LTV":        round(ltv_score, 2),
            "Sentiment":  round(sent_score, 2),
            "Risk Flags": round(flag_score, 2),
        })

    heat_df  = pd.DataFrame(heat_data).set_index("Borrower")
    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=["#6BC99A", "#FFD166", "#FF6B6B"],
        zmin=0, zmax=1,
        title="Risk Heatmap — All Borrowers × All Dimensions",
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(coloraxis_showscale=True)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df, x="borrower", y="principal_mm", color="sector",
            title="Exposure by Borrower ($M)",
            labels={"principal_mm": "Principal ($M)",
                    "borrower": "Borrower"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df, x="net_leverage", y="interest_coverage",
            color="sector", size="principal_mm",
            hover_name="borrower",
            title="Leverage vs Coverage Map",
        )
        fig2.add_hline(y=2.0, line_dash="dash", line_color="red",
                       annotation_text="Coverage floor 2x")
        fig2.add_vline(x=5.0, line_dash="dash", line_color="red",
                       annotation_text="Leverage limit 5x")
        st.plotly_chart(fig2, use_container_width=True)

    # ── RADAR COMPARISON ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Borrower Radar Comparison")
    col1, col2 = st.columns(2)
    b1 = col1.selectbox("Borrower A", df["borrower"].tolist(), index=1)
    b2 = col2.selectbox("Borrower B", df["borrower"].tolist(), index=2)

    fig_radar = go.Figure()
    for bname, colour in [(b1, "#4E9FD4"), (b2, "#E8B84B")]:
        row = df[df["borrower"] == bname].iloc[0]
        vals = radar_values(row)
        vals += [vals[0]]
        cats_closed = cats + [cats[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats_closed, fill="toself",
            name=bname, line_color=colour, opacity=0.7
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Risk Profile Comparison (higher = more risk)",
        showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — RISK WATCHLIST
# ══════════════════════════════════════════════════════════════════
elif "Watchlist" in page:
    st.title("Risk Watchlist")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Flagged",     int((df['risk_flags'] > 0).sum()))
    c2.metric("Covenant Breaches", int(df['covenant_breached'].sum()))
    c3.metric("Low Coverage",      int(df['low_coverage'].sum()))

    st.markdown("---")
    watchlist = df[df["risk_flags"] > 0].sort_values(
        "risk_flags", ascending=False
    )

    if watchlist.empty:
        st.success("No loans currently flagged.")
    else:
        st.warning(f"{len(watchlist)} loan(s) require attention")

        display_cols = [
            "borrower", "sector", "risk_flags",
            "covenant_breached", "net_leverage",
            "interest_coverage", "ltv_pct", "sentiment_score"
        ]

        def colour_row(row):
            c = "#FFCCCC" if row["risk_flags"] >= 3 \
                else "#FFE8CC" if row["risk_flags"] >= 2 \
                else "#FFFACC"
            return [f"background-color:{c}"] * len(row)

        st.dataframe(
            watchlist[display_cols].style.apply(colour_row, axis=1),
            use_container_width=True
        )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                watchlist, x="borrower", y="risk_flags",
                color="risk_flags",
                color_continuous_scale=["#FFFACC","#FFE8CC","#FF6B6B"],
                title="Risk Flags by Borrower"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Radar for all stressed borrowers
            fig_r = go.Figure()
            colours = ["#FF6B6B","#FFD166","#E8B84B","#F4A261","#E76F51"]
            for i, (_, row) in enumerate(watchlist.iterrows()):
                vals = radar_values(row)
                vals += [vals[0]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=cats + [cats[0]],
                    fill="toself", name=row["borrower"],
                    line_color=colours[i % len(colours)],
                    opacity=0.6
                ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(range=[0,1])),
                title="Stressed Borrower Risk Profiles"
            )
            st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — VALUATION ENGINE
# ══════════════════════════════════════════════════════════════════
elif "Valuation" in page:
    st.title("Valuation Engine")
    st.caption("ASC 820 / IFRS 13 · Build-up discount rate · DCF · Comps · Monte Carlo")

    col1, col2 = st.columns([2, 1])
    borrower   = col1.selectbox("Borrower", df["borrower"].tolist())
    instrument = col2.selectbox("Instrument", [
        "direct_lending","unitranche","second_lien",
        "mezzanine","pik","asset_based"
    ])
    row = df[df["borrower"] == borrower].iloc[0]

    # ── INTERACTIVE DISCOUNT RATE BUILDER ────────────────────────
    st.markdown("---")
    st.subheader("Interactive Discount Rate Builder")
    st.caption("Adjust any component — NAV updates in real time")

    base_rate = build_discount_rate(row, instrument)

    sc1, sc2, sc3 = st.columns(3)
    sofr_s  = sc1.slider("SOFR base (%)",      2.0, 8.0,
                          float(base_rate["sofr_base"]),     0.05)
    cs_s    = sc2.slider("Credit spread (%)",   0.5, 10.0,
                          float(base_rate["credit_spread"]), 0.05)
    ill_s   = sc3.slider("Illiquidity prem (%)",0.5, 4.0,
                          float(base_rate["illiquidity_prem"]), 0.05)

    sc4, sc5, sc6 = st.columns(3)
    cx_s    = sc4.slider("Complexity prem (%)", 0.0, 2.0,
                          float(base_rate["complexity_prem"]), 0.05)
    sec_s   = sc5.slider("Sector adj (%)",     -0.5, 1.5,
                          float(base_rate["sector_adj"]),    0.05)
    cov_s   = sc6.slider("Covenant adj (%)",    0.0, 2.0,
                          float(base_rate["covenant_adj"]),  0.05)

    live_rate = sofr_s + cs_s + ill_s + cx_s + sec_s + cov_s
    live_nav  = dcf_valuation(row, live_rate)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live Discount Rate", f"{live_rate:.2f}%")
    m2.metric("Live DCF NAV",       f"${live_nav['nav']:.2f}M")
    m3.metric("NAV / Par",          f"{live_nav['nav_to_par']:.1f}%")
    m4.metric("Duration",           f"{live_nav['duration_yrs']:.2f}yr")

    # Waterfall of components
    comps_names = ["SOFR","Credit spread","Illiquidity",
                   "Complexity","Sector adj","Covenant adj"]
    comps_vals  = [sofr_s, cs_s, ill_s, cx_s, sec_s, cov_s]
    fig_wf = go.Figure(go.Bar(
        x=comps_names, y=comps_vals,
        marker_color=["#4E9FD4","#E8B84B","#6BC99A",
                      "#9B8EA0","#D46E6E","#D46E6E"],
        text=[f"{v:.2f}%" for v in comps_vals],
        textposition="outside"
    ))
    fig_wf.add_hline(
        y=live_rate, line_dash="dash", line_color="red",
        annotation_text=f"Total: {live_rate:.2f}%"
    )
    fig_wf.update_layout(
        title="Discount Rate Components (drag sliders to update)",
        yaxis_title="Rate (%)", showlegend=False
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("---")

    # ── FULL VALUATION ────────────────────────────────────────────
    if st.button("Run Full Valuation", type="primary"):
        with st.spinner("DCF · Comps · Monte Carlo · Z-score..."):
            val = full_valuation(row, instrument)
        st.session_state[f"val_{borrower}"] = val

    if f"val_{borrower}" in st.session_state:
        val = st.session_state[f"val_{borrower}"]
        r   = val["rate_components"]
        d   = val["dcf"]
        c   = val["comps"]
        m   = val["monte_carlo"]
        z   = val["credit_score"]

        st.success(
            f"**{val['nav_flag']}** — Blended NAV ${val['blended_nav_mm']}M "
            f"| {val['asc820_level']} | {val['methodology_note']}"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Blended NAV",  f"${val['blended_nav_mm']}M",
                    f"{val['nav_to_par_pct']}% of par")
        col2.metric("Discount Rate", f"{r['total_rate_pct']}%")
        col3.metric("Z-Score",       f"{z['z_score']}",
                    z['zone'])
        col4.metric("Default Prob",  f"{m['pd_applied']}%")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("DCF Cash Flow Analysis")
            fig_dcf = go.Figure()
            fig_dcf.add_bar(
                x=[f"Yr {cf['period']}" for cf in d["cash_flows"]],
                y=[cf["cash_flow"]      for cf in d["cash_flows"]],
                name="Cash Flow", marker_color="#4E9FD4"
            )
            fig_dcf.add_bar(
                x=[f"Yr {cf['period']}" for cf in d["cash_flows"]],
                y=[cf["present_value"]  for cf in d["cash_flows"]],
                name="Present Value", marker_color="#E8B84B"
            )
            fig_dcf.update_layout(
                title=f"DCF NAV ${d['nav']}M — {d['nav_to_par']}% of par",
                barmode="group"
            )
            st.plotly_chart(fig_dcf, use_container_width=True)
            st.caption(
                f"Duration: {d['duration_yrs']}yr · "
                f"DV01: ${d['dv01_mm']}M per 100bps"
            )

        with col2:
            st.subheader("Monte Carlo Fan Chart")
            stress = st.toggle("Double default probability (stress scenario)")
            mc_data = val["monte_carlo"]

            if stress:
                with st.spinner("Re-running stress scenario..."):
                    row_stress = row.copy()
                    row_stress["risk_flags"] = min(int(row["risk_flags"]) + 2, 4)
                    from valuation import monte_carlo_valuation
                    mc_stress = monte_carlo_valuation(row_stress, instrument)
                navs_plot = np.array(mc_stress["nav_distribution"])
                p10_s, p50_s, p90_s = mc_stress["p10"], mc_stress["p50"], mc_stress["p90"]
                pd_label = f"Stress PD: {mc_stress['pd_applied']}%"
            else:
                navs_plot = np.array(mc_data["nav_distribution"])
                p10_s, p50_s, p90_s = mc_data["p10"], mc_data["p50"], mc_data["p90"]
                pd_label = f"Base PD: {mc_data['pd_applied']}%"

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=navs_plot, nbinsx=60,
                marker_color="#6BC99A" if not stress else "#FF6B6B",
                opacity=0.7, name="NAV distribution"
            ))
            for label, val_p, color in [
                ("P10", p10_s, "red"),
                ("P50", p50_s, "orange"),
                ("P90", p90_s, "green")
            ]:
                fig_mc.add_vline(
                    x=val_p, line_dash="dash", line_color=color,
                    annotation_text=f"{label}: ${val_p}M"
                )
            fig_mc.update_layout(
                title=f"Monte Carlo 10,000 sims · {pd_label}",
                xaxis_title="NAV ($M)", yaxis_title="Frequency"
            )
            st.plotly_chart(fig_mc, use_container_width=True)
            st.caption(
                f"P10: ${p10_s}M · P50: ${p50_s}M · P90: ${p90_s}M · "
                f"Expected Loss: ${mc_data['expected_loss_mm']}M"
            )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Comps Analysis")
            comps_df = pd.DataFrame({
                "Metric": ["Sector Multiple","Implied EV",
                           "Debt Coverage","Equity Cushion",
                           "Bear EV","Bull EV"],
                "Value":  [
                    f"{c['sector_multiple']}x EV/EBITDA",
                    f"${c['implied_ev']}M",
                    f"{c['ev_coverage']}x",
                    f"${c['equity_cushion_mm']}M ({c['equity_cushion_pct']}%)",
                    f"${c['ev_bear_case']}M",
                    f"${c['ev_bull_case']}M",
                ]
            })
            st.dataframe(comps_df, hide_index=True,
                         use_container_width=True)
            st.caption(c["coverage_assessment"])

        with col2:
            st.subheader("Credit Score — Altman Z Breakdown")
            z_comp = z["components"]
            fig_z = go.Figure(go.Bar(
                x=list(z_comp.keys()),
                y=list(z_comp.values()),
                marker_color=["#4E9FD4","#6BC99A","#E8B84B",
                              "#9B8EA0","#D46E6E"],
                text=[f"{v:.3f}" for v in z_comp.values()],
                textposition="outside"
            ))
            fig_z.update_layout(
                title=f"Z-Score: {z['z_score']} — {z['zone']}",
                yaxis_title="Component Score"
            )
            st.plotly_chart(fig_z, use_container_width=True)

        if val.get("recovery"):
            st.markdown("---")
            st.error("Covenant Breach — Recovery Analysis Triggered")
            rec = val["recovery"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Recovery Rate",  f"{rec['recovery_rate']*100:.1f}%")
            r2.metric("Recovery Value", f"${rec['recovery_value_mm']}M")
            r3.metric("LGD",            f"{rec['loss_given_default']*100:.1f}%")
            r4.metric("Asset Coverage", f"{rec['asset_coverage_ratio']}x")


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — CREDIT MEMO GENERATOR
# ══════════════════════════════════════════════════════════════════
elif "Memo" in page:
    st.title("AI Credit Memo Generator")
    st.caption("Institutional credit narratives grounded in full valuation stack")

    col1, col2 = st.columns([2, 1])
    borrower   = col1.selectbox("Borrower", df["borrower"].tolist())
    instrument = col2.selectbox("Instrument", [
        "direct_lending","unitranche","second_lien",
        "mezzanine","pik","asset_based"
    ])
    row   = df[df["borrower"] == borrower].iloc[0]
    flags = int(row["risk_flags"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Leverage",     f"{row['net_leverage']}x",
              delta=f"limit {row['leverage_limit']}x",
              delta_color="inverse")
    c2.metric("Interest Coverage",f"{row['interest_coverage']}x")
    c3.metric("LTV",              f"{row['ltv_pct']}%")
    c4.metric("Sentiment",        f"{row['sentiment_score']}")

    if flags >= 3:   st.error(f"HIGH RISK — {flags}/4 flags")
    elif flags >= 1: st.warning(f"WATCH — {flags}/4 flags")
    else:            st.success("HEALTHY — No flags")

    st.markdown("---")

    if st.button("Generate Credit Memo", type="primary"):
        with st.spinner("Running valuation + generating memo..."):
            val  = full_valuation(row, instrument)
            memo = generate_credit_narrative(row, val)

        st.session_state[f"memo_{borrower}"]   = memo
        st.session_state[f"memoval_{borrower}"] = val

    if f"memo_{borrower}" in st.session_state:
        memo = st.session_state[f"memo_{borrower}"]
        val  = st.session_state[f"memoval_{borrower}"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Blended NAV",   f"${val['blended_nav_mm']}M")
        col2.metric("Discount Rate", f"{val['rate_components']['total_rate_pct']}%")
        col3.metric("Z-Score",       f"{val['credit_score']['z_score']}")

        st.markdown("---")
        st.subheader("Credit Memo")

        # Parse and render sections with colour coding
        section_colours = {
            "HEALTH SUMMARY":       ("#EBF5FF", "#0C447C"),
            "VALUATION ASSESSMENT": ("#F0FFF4", "#0F6E56"),
            "KEY RISKS":            ("#FFF8E6", "#633806"),
            "RECOMMENDED ACTION":   ("#F0FFF4", "#0F6E56"),
        }

        current_section = None
        current_content = []

        for line in memo.split("\n"):
            matched = False
            for section, colours in section_colours.items():
                if section in line.upper():
                    # Flush previous
                    if current_section and current_content:
                        section_box(
                            current_section,
                            " ".join(current_content),
                            section_colours[current_section][0],
                            section_colours[current_section][1]
                        )
                    current_section = section
                    current_content = []
                    matched = True
                    break
            if not matched and line.strip():
                current_content.append(line.strip())

        # Flush last section
        if current_section and current_content:
            section_box(
                current_section,
                " ".join(current_content),
                section_colours[current_section][0],
                section_colours[current_section][1]
            )

        st.markdown("---")
        st.download_button(
            "Download memo (.txt)", memo,
            file_name=f"{borrower}_credit_memo.txt"
        )


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — ASK THE AI
# ══════════════════════════════════════════════════════════════════
elif "Ask" in page:
    st.title("Ask the AI")
    st.caption("Ask any question about your portfolio — grounded in live valuation data")

    borrower = st.selectbox(
        "Select borrower context", ["All borrowers"] + df["borrower"].tolist()
    )

    if "ask_q" not in st.session_state:
        st.session_state["ask_q"] = ""

    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)

    if col1.button("Why is BetaCorp below par?"):
        st.session_state["ask_q"] = (
            "Why is BetaCorp trading at only 89.4% of par? "
            "Break down every factor — discount rate components, "
            "covenant breach, Z-score, Monte Carlo default probability, "
            "and recovery risk in detail."
        )
    if col2.button("What if SOFR drops 100bps?"):
        st.session_state["ask_q"] = (
            "If SOFR drops by exactly 100 basis points from 4.33% to 3.33%, "
            "calculate the new NAV for every borrower using DV01 "
            "and show the total portfolio impact in dollars."
        )
    if col3.button("Which borrower has worst recovery?"):
        st.session_state["ask_q"] = (
            "Which borrower has the worst recovery profile? "
            "Compare recovery rates, LGD, asset coverage, and default "
            "probability across all borrowers and rank them."
        )

    question = st.text_input(
        "Your question",
        value=st.session_state["ask_q"],
        placeholder="e.g. Why is ZetaLogistics at risk? What is the portfolio DV01?"
    )
    st.session_state["ask_q"] = question

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running valuation and analysing..."):

                # Build full valuation context
                if borrower == "All borrowers":
                    val_context = []
                    for _, r in df.iterrows():
                        v = full_valuation(r, "direct_lending")
                        rec_str = (
                            f"Recovery ${v['recovery']['recovery_value_mm']}M "
                            f"({round(v['recovery']['recovery_rate']*100,1)}% rate), "
                            f"LGD {round(v['recovery']['loss_given_default']*100,1)}%"
                            if v.get("recovery") else "No covenant breach"
                        )
                        val_context.append(
                            f"\n{r['borrower']} ({r['sector']}):\n"
                            f"  Blended NAV: ${v['blended_nav_mm']}M "
                            f"({v['nav_to_par_pct']}% of par) — {v['nav_flag']}\n"
                            f"  Discount rate: {v['rate_components']['total_rate_pct']}% "
                            f"[SOFR {v['rate_components']['sofr_base']}% + "
                            f"spread {v['rate_components']['credit_spread']}% + "
                            f"illiquidity {v['rate_components']['illiquidity_prem']}% + "
                            f"covenant adj {v['rate_components']['covenant_adj']}%]\n"
                            f"  DCF NAV: ${v['dcf']['nav']}M | "
                            f"DV01: ${v['dcf']['dv01_mm']}M per 100bps | "
                            f"Duration: {v['dcf']['duration_yrs']}yr\n"
                            f"  Monte Carlo — P10: ${v['monte_carlo']['p10']}M | "
                            f"P50: ${v['monte_carlo']['p50']}M | "
                            f"P90: ${v['monte_carlo']['p90']}M | "
                            f"PD: {v['monte_carlo']['pd_applied']}% | "
                            f"Expected Loss: ${v['monte_carlo']['expected_loss_mm']}M\n"
                            f"  Comps EV: ${v['comps']['implied_ev']}M | "
                            f"Coverage: {v['comps']['ev_coverage']}x | "
                            f"Equity cushion: ${v['comps']['equity_cushion_mm']}M\n"
                            f"  Z-score: {v['credit_score']['z_score']} "
                            f"({v['credit_score']['zone']})\n"
                            f"  Recovery: {rec_str}\n"
                            f"  Leverage: {r['net_leverage']}x "
                            f"(limit {r['leverage_limit']}x) | "
                            f"Covenant breached: {r['covenant_breached']} | "
                            f"Coverage: {r['interest_coverage']}x | "
                            f"LTV: {r['ltv_pct']}% | "
                            f"Sentiment: {r['sentiment_score']}\n"
                            f"  News: {r['recent_news']}"
                        )
                    context_str = "\n".join(val_context)

                else:
                    row = df[df["borrower"] == borrower].iloc[0]
                    v   = full_valuation(row, "direct_lending")
                    r   = v["rate_components"]
                    rec_str = (
                        f"Recovery value ${v['recovery']['recovery_value_mm']}M, "
                        f"recovery rate {round(v['recovery']['recovery_rate']*100,1)}%, "
                        f"LGD {round(v['recovery']['loss_given_default']*100,1)}%, "
                        f"asset coverage {v['recovery']['asset_coverage_ratio']}x"
                        if v.get("recovery") else "No covenant breach detected"
                    )
                    context_str = f"""
BORROWER: {borrower} ({row['sector']})

VALUATION SUMMARY
  Blended NAV: ${v['blended_nav_mm']}M ({v['nav_to_par_pct']}% of par) — {v['nav_flag']}
  Weighting: {v['weighting']}
  ASC 820: {v['asc820_level']}

DISCOUNT RATE BUILD-UP: {r['total_rate_pct']}% total
  SOFR base:          {r['sofr_base']}%
  Credit spread:      {r['credit_spread']}% [{r['credit_bucket']} bucket]
  Illiquidity prem:   {r['illiquidity_prem']}%
  Complexity prem:    {r['complexity_prem']}%
  Sector adjustment:  {r['sector_adj']}%
  Covenant adjustment:{r['covenant_adj']}%
  BSL premium:        {r['bsl_premium_pct']}%

DCF ANALYSIS
  NAV: ${v['dcf']['nav']}M ({v['dcf']['nav_to_par']}% of par)
  Duration: {v['dcf']['duration_yrs']} years
  DV01: ${v['dcf']['dv01_mm']}M per 100bps rate move

MONTE CARLO (10,000 simulations)
  P10: ${v['monte_carlo']['p10']}M | P25: ${v['monte_carlo']['p25']}M
  P50: ${v['monte_carlo']['p50']}M | P75: ${v['monte_carlo']['p75']}M
  P90: ${v['monte_carlo']['p90']}M
  Default probability: {v['monte_carlo']['pd_applied']}%
  Expected loss: ${v['monte_carlo']['expected_loss_mm']}M

COMPS
  Sector multiple: {v['comps']['sector_multiple']}x EV/EBITDA
  Implied EV: ${v['comps']['implied_ev']}M
  Debt coverage: {v['comps']['ev_coverage']}x
  Equity cushion: ${v['comps']['equity_cushion_mm']}M ({v['comps']['equity_cushion_pct']}%)
  Bear case EV: ${v['comps']['ev_bear_case']}M | Bull: ${v['comps']['ev_bull_case']}M
  Assessment: {v['comps']['coverage_assessment']}

CREDIT SCORE
  Z-score: {v['credit_score']['z_score']} — {v['credit_score']['zone']}

RECOVERY ANALYSIS
  {rec_str}

RISK SIGNALS
  Net leverage: {row['net_leverage']}x (covenant limit: {row['leverage_limit']}x)
  Covenant breached: {row['covenant_breached']}
  Interest coverage: {row['interest_coverage']}x
  LTV: {row['ltv_pct']}%
  Sentiment score: {row['sentiment_score']}
  Risk flags: {int(row['risk_flags'])} of 4
  Recent news: {row['recent_news']}
"""

                prompt = f"""You are a senior private credit analyst at an institutional asset manager.

CRITICAL RULES:
1. You MUST reference specific numbers from the data in every sentence.
2. Never make a general statement without backing it with a number.
3. Be direct and analytical — write like you are presenting to a CIO.
4. Structure your answer clearly. Use numbers, percentages, and dollar amounts throughout.
5. Minimum 4-5 sentences. Maximum 8 sentences.
6. Do not start with "Based on the data" or any preamble. Start with the answer directly.

Portfolio data:
{context_str}

Question: {question}

Answer with specific numbers:"""

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content

            st.markdown("---")
            st.markdown(f"**Q: {question}**")
            st.info(answer)
# ══════════════════════════════════════════════════════════════════
# PAGE 6 — NAV SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif "NAV" in page:
    st.title("NAV Scenario Analysis")
    st.caption("Real-time portfolio NAV sensitivity to discount rate")

    rate = st.slider(
        "Discount Rate (%)", 4.0, 20.0, 10.0, step=0.25,
        format="%.2f%%"
    )

    def loan_nav(row, dr):
        dr_d = dr / 100
        p    = row["principal_mm"]
        c    = p * row["coupon_pct"] / 100
        cf   = [c] * 4; cf[-1] += p
        return sum(v / (1 + dr_d) ** t for t, v in enumerate(cf, 1))

    df["scenario_nav"] = df.apply(lambda r: loan_nav(r, rate), axis=1)
    total_nav = df["scenario_nav"].sum()
    total_par = df["principal_mm"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio NAV",  f"${total_nav:.1f}M")
    c2.metric("Par Value",      f"${total_par:.0f}M")
    c3.metric("NAV vs Par",     f"${total_nav - total_par:.1f}M")
    c4.metric("Avg NAV/Par",    f"{total_nav/total_par*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df, x="borrower", y="scenario_nav", color="sector",
            title=f"Loan NAV at {rate:.2f}% Discount Rate ($M)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rates = np.arange(4.0, 20.25, 0.25)
        navs  = [df.apply(lambda r: loan_nav(r, dr), axis=1).sum()
                 for dr in rates]
        fig2  = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rates, y=navs, mode="lines",
            line=dict(color="#4E9FD4", width=2),
            fill="tozeroy", fillcolor="rgba(78,159,212,0.1)"
        ))
        fig2.add_vline(
            x=rate, line_dash="dash", line_color="red",
            annotation_text=f"Current: {rate:.2f}%"
        )
        fig2.add_hline(
            y=total_par, line_dash="dot", line_color="gray",
            annotation_text="Par value"
        )
        fig2.update_layout(
            title="Portfolio NAV Sensitivity Curve",
            xaxis_title="Discount Rate (%)",
            yaxis_title="Portfolio NAV ($M)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # DV01 breakdown per loan
    st.markdown("---")
    st.subheader("DV01 — Dollar Sensitivity per Loan")
    st.caption("$ change in NAV per 100bps rate increase")
    dv01_data = []
    for _, row in df.iterrows():
        nav_base = loan_nav(row, rate)
        nav_up   = loan_nav(row, rate + 1.0)
        dv01_data.append({
            "borrower": row["borrower"],
            "dv01_mm":  round(nav_base - nav_up, 3)
        })
    dv01_df = pd.DataFrame(dv01_data).sort_values("dv01_mm")
    fig_dv = px.bar(
        dv01_df, x="borrower", y="dv01_mm",
        title="DV01 by Borrower ($M per 100bps)",
        color="dv01_mm",
        color_continuous_scale=["#6BC99A", "#FFD166", "#FF6B6B"]
    )
    st.plotly_chart(fig_dv, use_container_width=True)