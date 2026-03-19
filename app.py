import streamlit as st
from data import load_portfolio
from valuation import full_valuation
from narrative import generate_credit_narrative

st.set_page_config(page_title="Private Credit Intelligence", layout="wide")

st.title("📊 Private Credit Intelligence Platform")

# ── LOAD DATA ────────────────────────────────────────────
df = load_portfolio()

if df.empty:
    st.error("No data available")
    st.stop()

# ── SELECT BORROWER ──────────────────────────────────────
borrower = st.selectbox("Select Borrower", df["borrower"].unique())

row = df[df["borrower"] == borrower].iloc[0]

# ── VALUATION ────────────────────────────────────────────
val = full_valuation(row, "direct_lending")

# ── METRICS ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

col1.metric("Blended NAV", f"${val['blended_nav_mm']}M")
col2.metric("Discount Rate", f"{val['rate_components']['total_rate_pct']}%")
col3.metric("Z-Score", f"{val['credit_score']['z_score']}")

st.divider()

# ── GENERATE MEMO BUTTON ─────────────────────────────────
if st.button("Generate Credit Memo"):

    with st.spinner("Generating memo..."):
        memo = generate_credit_narrative(row, val)

    st.subheader("📄 Credit Memo")

    # ── ERROR HANDLING ───────────────────────────────────
    if memo.startswith("Error"):
        st.error(memo)
    else:
        st.success("Memo generated successfully")

        st.markdown(
            f"""
            <div style="
                background-color:#f9f9f9;
                padding:20px;
                border-radius:10px;
                border:1px solid #eee;
                font-size:15px;
                line-height:1.6;
            ">
            {memo}
            </div>
            """,
            unsafe_allow_html=True
        )