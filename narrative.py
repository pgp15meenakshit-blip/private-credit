from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_credit_narrative(row, valuation_result=None):

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
  Sentiment Score:   {row['sentiment_score']}  (-1.0 negative, +1.0 positive)
  Recent News:       {row['recent_news']}
  Risk Flags:        {int(row['risk_flags'])} of 4 triggered
"""

    val_data = ""
    if valuation_result:
        r   = valuation_result["rate_components"]
        d   = valuation_result["dcf"]
        c   = valuation_result["comps"]
        m   = valuation_result["monte_carlo"]
        z   = valuation_result["credit_score"]
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
  Z-score:  {z['z_score']}  ->  {z['zone']}
"""
        if rec:
            val_data += f"""
RECOVERY ANALYSIS  (Covenant breach triggered)
  Recovery rate:  {rec['recovery_rate']*100:.1f}%
  Recovery value: ${rec['recovery_value_mm']}M
  LGD:            {rec['loss_given_default']*100:.1f}%
  Asset coverage: {rec['asset_coverage_ratio']}x
"""

    has_val = valuation_result is not None
    prompt = f"""You are a senior private credit analyst at an institutional asset manager.
Write a professional credit memo in exactly 4 labelled sections.
Use these exact section headers on their own line:

HEALTH SUMMARY
VALUATION ASSESSMENT
KEY RISKS
RECOMMENDED ACTION

Rules:
- Reference specific numbers in every sentence. Never be vague.
- HEALTH SUMMARY: overall credit health, key ratios, blended NAV vs par, ASC 820 level.
- VALUATION ASSESSMENT: DCF NAV vs par, comps EV coverage, Monte Carlo P10/P50/P90, DV01 impact, blended NAV weighting rationale.
- KEY RISKS: exactly 3 risks in order of severity, each with specific numbers. Flag Z-score zone and covenant breach explicitly if present.
- RECOMMENDED ACTION: one clear action (Hold / Watchlist / Request cure / Enforce covenant / Restructure / Exit). Include timeline. Justify using recovery analysis if breach detected.
- Write as a professional analyst. No preamble. Start directly with HEALTH SUMMARY.

{base_data}
{val_data}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    from data import load_portfolio
    from valuation import full_valuation
    df = load_portfolio()
    for borrower in ["BetaCorp", "GammaTech"]:
        row = df[df["borrower"] == borrower].iloc[0]
        val = full_valuation(row, "direct_lending")
        print(f"\n{'='*60}\n  CREDIT MEMO — {borrower}\n{'='*60}\n")
        print(generate_credit_narrative(row, val))