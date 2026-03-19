from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ── SECURE CLIENT INITIALIZATION ────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not set. Please configure environment variable.")

client = Groq(api_key=api_key)


def generate_credit_narrative(row, valuation_result=None):
    """
    Generates a structured AI credit memo.
    """

    # ── BASE BORROWER SIGNALS ─────────────────────────────────────
    base_data = f"""
BORROWER PROFILE
  Borrower:          {row['borrower']}
  Sector:            {row['sector']}
  Principal:         ${row['principal_mm']}M
  Coupon:            {row['coupon_pct']}%
  Maturity:          {row['maturity_date']}
  Net Leverage:      {row['net_leverage']}x  (covenant limit: {row['leverage_limit']}x)
  Covenant Breached: {row['covenant_breached']}
  Interest Coverage: {row['interest_coverage']}x
  LTV:               {row['ltv_pct']}%
  EBITDA:            ${row['ebitda_mm']}M
  Sentiment Score:   {row['sentiment_score']}
  Recent News:       {row['recent_news']}
  Risk Flags:        {int(row['risk_flags'])} of 4 triggered
"""

    # ── VALUATION BLOCK ───────────────────────────────────────────
    val_data = ""
    if valuation_result:
        r = valuation_result["rate_components"]
        d = valuation_result["dcf"]
        c = valuation_result["comps"]
        m = valuation_result["monte_carlo"]
        z = valuation_result["credit_score"]

        val_data = f"""
DISCOUNT RATE BUILD-UP
  Total discount rate: {r['total_rate_pct']}%

VALUATION OUTPUTS
  DCF NAV:        ${d['nav']}M ({d['nav_to_par']}% of par)
  Duration:       {d['duration_yrs']} years
  DV01:           ${d['dv01_mm']}M
  Comps EV:       ${c['implied_ev']}M
  Equity cushion: ${c['equity_cushion_mm']}M

MONTE CARLO
  P10: ${m['p10']}M | P50: ${m['p50']}M | P90: ${m['p90']}M
  Expected loss: ${m['expected_loss_mm']}M

CREDIT SCORE
  Z-score: {z['z_score']} → {z['zone']}

BLENDED NAV: ${valuation_result['blended_nav_mm']}M
"""

    # ── PROMPT ────────────────────────────────────────────────────
    prompt = f"""
You are a senior private credit analyst.

Write a structured credit memo with these sections:

HEALTH SUMMARY  
VALUATION ASSESSMENT  
KEY RISKS  
RECOMMENDED ACTION  

Be precise, use numbers, and sound like an investment professional.

{base_data}
{val_data}
"""

    # ── GROQ API CALL (FIXED) ─────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a credit analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=900,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating narrative: {str(e)}"


# ── TEST ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data import load_portfolio
    from valuation import full_valuation

    df = load_portfolio()

    for borrower in ["BetaCorp", "GammaTech"]:
        row = df[df["borrower"] == borrower].iloc[0]
        val = full_valuation(row, "direct_lending")

        print(f"\n{'='*60}")
        print(f"CREDIT MEMO — {borrower}")
        print(f"{'='*60}\n")

        memo = generate_credit_narrative(row, val)
        print(memo)
        print()