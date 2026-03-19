from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("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"))


def generate_credit_narrative(row, valuation_result=None):
    """
    Generates a structured AI credit memo.
    If valuation_result is passed, the memo references full
    valuation output — discount rate, DCF, comps, Monte Carlo,
    Z-score, and recovery analysis.
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
  Sentiment Score:   {row['sentiment_score']}  (-1.0 = very negative, +1.0 = very positive)
  Recent News:       {row['recent_news']}
  Risk Flags:        {int(row['risk_flags'])} of 4 triggered
"""

    # ── VALUATION BLOCK (if available) ───────────────────────────
    val_data = ""
    if valuation_result:
        r  = valuation_result["rate_components"]
        d  = valuation_result["dcf"]
        c  = valuation_result["comps"]
        m  = valuation_result["monte_carlo"]
        z  = valuation_result["credit_score"]
        rec = valuation_result.get("recovery")

        val_data = f"""
DISCOUNT RATE BUILD-UP  ({r['methodology']})
  SOFR base:            {r['sofr_base']}%
  Credit spread:        {r['credit_spread']}%  [{r['credit_bucket']} bucket]
  Illiquidity premium:  {r['illiquidity_prem']}%
  Complexity premium:   {r['complexity_prem']}%
  Sector adjustment:    {r['sector_adj']}%
  Covenant adjustment:  {r['covenant_adj']}%
  Total discount rate:  {r['total_rate_pct']}%
  BSL market premium:   {r['bsl_premium_pct']}%
  ASC 820 level:        {r['asc820_level']}

VALUATION OUTPUTS
  DCF NAV:        ${d['nav']}M  ({d['nav_to_par']}% of par)
  Duration:       {d['duration_yrs']} years
  DV01:           ${d['dv01_mm']}M per 100bps rate move
  Comps EV:       ${c['implied_ev']}M  ({c['ev_coverage']}x debt coverage)
  Equity cushion: ${c['equity_cushion_mm']}M  ({c['equity_cushion_pct']}% of EV)
  Comps verdict:  {c['coverage_assessment']}

MONTE CARLO  ({m['n_simulations']:,} simulations)
  P10 (stress):   ${m['p10']}M
  P25:            ${m['p25']}M
  P50 (base):     ${m['p50']}M
  P75:            ${m['p75']}M
  P90 (upside):   ${m['p90']}M
  Expected loss:  ${m['expected_loss_mm']}M
  Default prob:   {m['pd_applied']}%

BLENDED NAV:  ${valuation_result['blended_nav_mm']}M
  Weighting:  {valuation_result['weighting']}
  Assessment: {valuation_result['nav_flag']}

CREDIT SCORE  (Adapted Altman Z)
  Z-score:  {z['z_score']}  →  {z['zone']}
"""
        if rec:
            val_data += f"""
RECOVERY ANALYSIS  (Covenant breach triggered)
  Recovery rate:  {rec['recovery_rate']*100:.1f}%
  Recovery value: ${rec['recovery_value_mm']}M
  LGD:            {rec['loss_given_default']*100:.1f}%
  Asset coverage: {rec['asset_coverage_ratio']}x
"""

    # ── PROMPT ────────────────────────────────────────────────────
    has_valuation = valuation_result is not None
    prompt = f"""You are a senior private credit analyst at an alternative asset manager.
Your role is to write institutional-grade credit memos for the portfolio management team.

{"You have full quantitative valuation data available. Use it extensively." if has_valuation else "Write based on the available risk signals."}

Write a credit memo in exactly 4 sections using these headers:

**HEALTH SUMMARY**
Assess the overall credit profile. Reference leverage, coverage, LTV, and sentiment.
{"Reference the blended NAV, discount rate, and what ASC 820 level this sits at." if has_valuation else ""}
State clearly whether the borrower is healthy, on watch, or in distress.

**VALUATION ASSESSMENT**
{"Discuss the DCF NAV vs par, the comps EV coverage, and the Monte Carlo distribution. Explain why P10 and P50 differ. Reference DV01 — what a 100bps rate move means in dollar terms. State the blended NAV and weighting rationale." if has_valuation else "Estimate relative valuation based on available metrics."}

**KEY RISKS**
Identify the 3 most material risks in order of severity.
Be specific — reference exact numbers, not generalities.
{"If Z-score is in distress zone, flag workout risk explicitly. If covenant is breached, state the exact breach and legal implications." if has_valuation else ""}
Include sentiment signal and what the recent news implies for credit trajectory.

**RECOMMENDED ACTION**
Give ONE clear, specific action for the portfolio manager.
Choose from: Hold / Watchlist / Request cure / Enforce covenant rights /
             Initiate restructuring / Exit position.
{"Justify using the recovery analysis if breach is detected." if has_valuation else ""}
Include a suggested timeline (e.g. 'within 5 business days').

RULES:
- Reference exact numbers throughout. Never be vague.
- Write as a professional analyst, not an AI assistant.
- Be direct and decisive. Portfolio managers need clear guidance.
- Do not add any preamble or sign-off. Start directly with HEALTH SUMMARY.

{base_data}
{val_data}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ── TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data import load_portfolio
    from valuation import full_valuation

    df = load_portfolio()

    for borrower in ["BetaCorp", "GammaTech"]:
        row = df[df["borrower"] == borrower].iloc[0]
        val = full_valuation(row, "direct_lending")
        print(f"\n{'='*60}")
        print(f"  CREDIT MEMO — {borrower}")
        print(f"{'='*60}\n")
        memo = generate_credit_narrative(row, val)
        print(memo)
        print()