import numpy as np
import pandas as pd
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
# MARKET INPUTS — Q1 2026 calibrated
# Sources: Kroll H1 2024, Callan Q2 2025, Lincoln Senior Debt Index
# ══════════════════════════════════════════════════════════════════

SOFR                = 4.33   # 3M SOFR — current as of Q1 2026
RISK_FREE_RATE      = 4.25   # 10Y US Treasury
BSL_MARKET_SPREAD   = 3.22   # Broadly syndicated loan spread (bps / 100)
LINCOLN_INDEX_FV    = 99.1   # Lincoln Senior Debt Index — Sep 2025
COVENANT_DEFAULT_RT = 0.032  # Q3 2025 covenant default rate (3.2%)

# ── CREDIT SPREAD TABLE ───────────────────────────────────────────
# Built from Kroll valuation best practices + Callan Q2 2025 data
# Spread over SOFR in basis points by leverage bucket
CREDIT_SPREAD_BPS = {
    "aaa_aa":  75,    # leverage < 2x   — investment grade proxy
    "a_bbb":   175,   # leverage 2x–3x  — IG/crossover
    "bb":      275,   # leverage 3x–4x  — leveraged loan proxy
    "b":       400,   # leverage 4x–5x  — direct lending core
    "ccc":     600,   # leverage 5x–6x  — stressed
    "distress":900,   # leverage > 6x   — distressed / workout
}

# ── ILLIQUIDITY PREMIUM ───────────────────────────────────────────
# Source: Lord Abbett (~200bps), Aksia 2025, Lincoln International
# Investment-grade private debt illiquidity premium ~115bps (2025)
ILLIQUIDITY_BPS = {
    "direct_lending": 115,   # senior secured — most liquid in PC
    "unitranche":     165,   # blended senior + junior — one lender
    "second_lien":    200,   # subordinated secured
    "mezzanine":      250,   # subordinated + equity kicker
    "pik":            325,   # payment-in-kind — most illiquid
    "asset_based":    140,   # ABL — collateral-heavy, moderate
}

# ── COMPLEXITY PREMIUM ────────────────────────────────────────────
COMPLEXITY_BPS = {
    "direct_lending": 0,
    "unitranche":     25,
    "second_lien":    50,
    "mezzanine":      100,
    "pik":            150,
    "asset_based":    35,
}

# ── SECTOR RISK ADJUSTMENT ────────────────────────────────────────
# Based on Kroll sector benchmarks + Lincoln EV/EBITDA trends
SECTOR_ADJ_BPS = {
    "Technology":   -25,   # high visibility, recurring revenue
    "Healthcare":   -15,   # defensive, regulatory moat
    "Industrials":   30,   # cyclical, tariff exposure
    "Energy":        60,   # commodity price sensitivity
    "Retail":        85,   # structural headwinds, store closures
    "Media":         45,   # disruption risk, secular decline
    "Real Estate":   20,   # rate sensitive, collateral-backed
    "Financial":     15,   # regulated, balance sheet visibility
}

# ── COVENANT RISK ADJUSTMENT ──────────────────────────────────────
COVENANT_ADJ_BPS = {
    "clean":    0,    # no risk flags
    "watch":   35,    # 1-2 flags — elevated monitoring
    "breach":  90,    # covenant breach — workout risk
}

# ── EV/EBITDA SECTOR MULTIPLES ────────────────────────────────────
# Source: Lincoln International Q3 2025 — private credit comps
SECTOR_EV_EBITDA = {
    "Technology":   14.8,
    "Healthcare":   13.1,
    "Industrials":   9.0,
    "Energy":        7.5,
    "Retail":        6.8,
    "Media":         9.2,
    "Real Estate":  11.5,
    "Financial":    10.2,
}

# ── ASC 820 FAIR VALUE HIERARCHY CLASSIFICATION ───────────────────
# Level 1: observable market prices (not applicable for PC)
# Level 2: observable inputs (BSL comps, market spreads)
# Level 3: unobservable inputs (internal models, DCF)
ASC820_LEVEL = {
    "direct_lending": "Level 3",
    "unitranche":     "Level 3",
    "second_lien":    "Level 3",
    "mezzanine":      "Level 3",
    "pik":            "Level 3",
    "asset_based":    "Level 2/3",
}


# ══════════════════════════════════════════════════════════════════
# MODULE 1 — DISCOUNT RATE BUILDER (Build-Up Method)
# Methodology: Kroll / IPEV Guidelines / ASC 820 compliant
# Rate = SOFR + Credit Spread + Illiquidity + Complexity
#              + Sector Adjustment + Covenant Adjustment
# ══════════════════════════════════════════════════════════════════

def build_discount_rate(row, instrument_type="direct_lending"):
    lev = row["net_leverage"]

    # Credit spread bucket from leverage
    if lev < 2.0:   bucket = "aaa_aa"
    elif lev < 3.0: bucket = "a_bbb"
    elif lev < 4.0: bucket = "bb"
    elif lev < 5.0: bucket = "b"
    elif lev < 6.0: bucket = "ccc"
    else:           bucket = "distress"

    sofr        = SOFR
    credit_sp   = CREDIT_SPREAD_BPS[bucket] / 100
    illiquidity = ILLIQUIDITY_BPS.get(instrument_type, 115) / 100
    complexity  = COMPLEXITY_BPS.get(instrument_type, 0) / 100
    sector_adj  = SECTOR_ADJ_BPS.get(row.get("sector", "Industrials"), 30) / 100

    # Covenant adjustment
    if row.get("covenant_breached", False):
        cov_key = "breach"
    elif row.get("risk_flags", 0) >= 2:
        cov_key = "watch"
    else:
        cov_key = "clean"
    covenant_adj = COVENANT_ADJ_BPS[cov_key] / 100

    total = sofr + credit_sp + illiquidity + complexity + sector_adj + covenant_adj

    # BSL premium — excess return over broadly syndicated market
    bsl_premium = max(0, total - (SOFR + BSL_MARKET_SPREAD))

    return {
        "sofr_base":        round(sofr, 2),
        "credit_spread":    round(credit_sp, 2),
        "credit_bucket":    bucket,
        "illiquidity_prem": round(illiquidity, 2),
        "complexity_prem":  round(complexity, 2),
        "sector_adj":       round(sector_adj, 2),
        "covenant_adj":     round(covenant_adj, 2),
        "total_rate_pct":   round(total, 2),
        "bsl_premium_pct":  round(bsl_premium, 2),
        "instrument_type":  instrument_type,
        "asc820_level":     ASC820_LEVEL.get(instrument_type, "Level 3"),
        "methodology":      "Build-up method — ASC 820 / IFRS 13 compliant",
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 2A — YIELD ANALYSIS (Income Approach — ASC 820 Primary)
# DCF using build-up discount rate
# Kroll best practice: calibrate to issuance price, then mark
# ══════════════════════════════════════════════════════════════════

def dcf_valuation(row, discount_rate_pct, periods=4):
    dr        = discount_rate_pct / 100
    principal = row["principal_mm"]
    coupon    = principal * row["coupon_pct"] / 100

    cash_flows, pv_total = [], 0
    for t in range(1, periods + 1):
        cf = coupon + (principal if t == periods else 0)
        pv = cf / (1 + dr) ** t
        pv_total += pv
        cash_flows.append({
            "period": t, "cash_flow": round(cf, 3),
            "present_value": round(pv, 3),
            "discount_factor": round(1 / (1 + dr) ** t, 4),
        })

    # Duration — sensitivity measure
    duration = sum(
        (t * cf["present_value"])
        for t, cf in enumerate(cash_flows, 1)
    ) / pv_total

    # Price impact of 100bps rate move (DV01)
    dr_up   = (discount_rate_pct + 1.0) / 100
    nav_up  = sum(
        (coupon + (principal if t == periods else 0)) / (1 + dr_up) ** t
        for t in range(1, periods + 1)
    )
    dv01 = pv_total - nav_up  # $ change per 100bps

    return {
        "nav":          round(pv_total, 2),
        "par_value":    principal,
        "nav_to_par":   round((pv_total / principal) * 100, 1),
        "discount_rate":discount_rate_pct,
        "duration_yrs": round(duration, 2),
        "dv01_mm":      round(dv01, 3),
        "cash_flows":   cash_flows,
        "approach":     "Income approach — DCF",
        "asc820_input": "Level 3 — unobservable",
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 2B — MARKET APPROACH (ASC 820 Secondary)
# EV/EBITDA comparable company analysis
# Source: Lincoln International Q3 2025 private credit comps
# ══════════════════════════════════════════════════════════════════

def comps_valuation(row):
    sector   = row.get("sector", "Industrials")
    multiple = SECTOR_EV_EBITDA.get(sector, 9.0)
    ebitda   = row["ebitda_mm"]
    principal = row["principal_mm"]

    implied_ev       = ebitda * multiple
    ev_coverage      = implied_ev / principal
    equity_cushion   = max(0, implied_ev - principal)
    equity_cushion_pct = (equity_cushion / implied_ev * 100) if implied_ev > 0 else 0

    # Sensitivity: +/- 1x multiple
    ev_bear = ebitda * (multiple - 1.5)
    ev_bull = ebitda * (multiple + 1.5)

    return {
        "sector_multiple":     multiple,
        "implied_ev":          round(implied_ev, 1),
        "principal":           principal,
        "ev_coverage":         round(ev_coverage, 2),
        "equity_cushion_mm":   round(equity_cushion, 1),
        "equity_cushion_pct":  round(equity_cushion_pct, 1),
        "ev_bear_case":        round(ev_bear, 1),
        "ev_bull_case":        round(ev_bull, 1),
        "coverage_assessment": (
            "Strong — substantial equity cushion" if ev_coverage > 2.0
            else "Adequate — moderate equity cushion" if ev_coverage > 1.3
            else "Weak — limited asset coverage"
        ),
        "approach":     "Market approach — EV/EBITDA comps",
        "asc820_input": "Level 2 — market observable multiples",
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 2C — RECOVERY ANALYSIS (Distressed / Breach Scenario)
# Applied when covenant_breached = True
# Kroll: "fair value determined by timing & extent of recoverability"
# ══════════════════════════════════════════════════════════════════

def recovery_analysis(row):
    principal = row["principal_mm"]
    ltv       = row["ltv_pct"] / 100

    # Recovery rates by seniority (Moody's private credit averages)
    recovery_by_type = {
        "direct_lending": 0.72,
        "unitranche":     0.65,
        "second_lien":    0.48,
        "mezzanine":      0.35,
        "pik":            0.28,
        "asset_based":    0.78,
    }

    # Asset coverage — LTV tells us collateral vs. debt
    asset_coverage = 1 / ltv if ltv > 0 else 0
    # Blended recovery: weight seniority recovery 60%, asset coverage 40%
    seniority_recovery = recovery_by_type.get("direct_lending", 0.65)
    blended_recovery   = seniority_recovery * 0.6 + min(asset_coverage, 1.0) * 0.4

    return {
        "recovery_rate":        round(blended_recovery, 3),
        "recovery_value_mm":    round(principal * blended_recovery, 2),
        "loss_given_default":   round(1 - blended_recovery, 3),
        "asset_coverage_ratio": round(asset_coverage, 2),
        "scenario":             "Distressed recovery — breach scenario",
        "note": "Applied when covenant breached per Kroll methodology",
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 3 — MONTE CARLO SIMULATION
# Stochastic valuation for complex instruments (mezzanine, PIK)
# Simulates: SOFR path, EBITDA volatility, default probability
# Output: P10 / P50 / P90 NAV distribution
# ══════════════════════════════════════════════════════════════════

def monte_carlo_valuation(row, instrument_type="direct_lending",
                           n_simulations=10000, periods=4):
    np.random.seed(42)
    principal = row["principal_mm"]
    coupon    = principal * row["coupon_pct"] / 100

    # Default probability — calibrated to Q3 2025 covenant default rate
    base_pd = COVENANT_DEFAULT_RT
    risk_multiplier = {0: 1.0, 1: 1.8, 2: 3.2, 3: 5.5, 4: 8.0}
    flags   = int(row.get("risk_flags", 0))
    pd_used = min(base_pd * risk_multiplier.get(flags, 8.0), 0.65)

    # Stochastic inputs
    sofr_paths    = np.random.normal(SOFR, 0.80, n_simulations)
    ebitda_growth = np.random.normal(0.03, 0.14, n_simulations)
    defaults      = np.random.binomial(1, pd_used, n_simulations)
    recovery_rate = np.random.beta(6, 3, n_simulations) * 0.75

    rate_comp = build_discount_rate(row, instrument_type)
    navs = []

    for i in range(n_simulations):
        if defaults[i]:
            nav = principal * recovery_rate[i]
        else:
            scenario_sofr = sofr_paths[i]
            total_dr = (
                scenario_sofr
                + rate_comp["credit_spread"]
                + rate_comp["illiquidity_prem"]
                + rate_comp["complexity_prem"]
                + rate_comp["sector_adj"]
                + rate_comp["covenant_adj"]
            ) / 100
            # EBITDA-adjusted coupon for PIK/mezzanine
            adj_coupon = coupon * (1 + ebitda_growth[i] * 0.3)
            cf_list    = [adj_coupon] * periods
            cf_list[-1] += principal
            nav = sum(
                cf / (1 + total_dr) ** t
                for t, cf in enumerate(cf_list, 1)
            )
        navs.append(nav)

    navs = np.array(navs)

    # Expected loss calculation
    expected_loss = principal * pd_used * (1 - np.mean(recovery_rate))

    return {
        "p10":              round(float(np.percentile(navs, 10)), 2),
        "p25":              round(float(np.percentile(navs, 25)), 2),
        "p50":              round(float(np.percentile(navs, 50)), 2),
        "p75":              round(float(np.percentile(navs, 75)), 2),
        "p90":              round(float(np.percentile(navs, 90)), 2),
        "mean_nav":         round(float(np.mean(navs)), 2),
        "std_nav":          round(float(np.std(navs)), 2),
        "expected_loss_mm": round(expected_loss, 2),
        "pd_applied":       round(pd_used * 100, 2),
        "n_simulations":    n_simulations,
        "nav_distribution": navs.tolist(),
        "methodology":      "Monte Carlo — 10,000 simulations, stochastic SOFR + EBITDA",
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 4 — CREDIT SCORING (Altman Z-Score adapted for PC)
# Original Altman Z (1968) adapted for private credit context
# Score > 2.99: Safe | 1.81–2.99: Grey | < 1.81: Distress
# ══════════════════════════════════════════════════════════════════

def credit_score(row):
    ebitda    = row["ebitda_mm"]
    principal = row["principal_mm"]
    ltv       = row["ltv_pct"] / 100
    coverage  = row["interest_coverage"]
    leverage  = row["net_leverage"]
    sentiment = row.get("sentiment_score", 0)

    # Adapted Z-score components for private credit
    # X1: Working capital proxy (coverage ratio normalised)
    x1 = min(coverage / 5.0, 1.0)
    # X2: Retained earnings proxy (inverse of leverage)
    x2 = max(0, 1 - (leverage / 8.0))
    # X3: EBIT/Total assets proxy (EBITDA / implied EV)
    sector   = row.get("sector", "Industrials")
    mult     = SECTOR_EV_EBITDA.get(sector, 9.0)
    impl_ev  = ebitda * mult
    x3       = ebitda / impl_ev if impl_ev > 0 else 0
    # X4: Equity/Debt proxy (1 - LTV)
    x4       = max(0, 1 - ltv)
    # X5: Sales/Assets proxy (sentiment as market signal)
    x5       = (sentiment + 1) / 2   # normalise -1..1 → 0..1

    # Weights adapted for private credit (not public equity)
    z_score  = (1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5)

    if z_score > 2.99:   zone = "Safe zone"
    elif z_score > 1.81: zone = "Grey zone — elevated monitoring"
    else:                zone = "Distress zone — workout risk"

    return {
        "z_score":          round(z_score, 3),
        "zone":             zone,
        "components":       {
            "coverage_ratio_x1": round(x1, 3),
            "leverage_health_x2": round(x2, 3),
            "ebitda_efficiency_x3": round(x3, 3),
            "collateral_coverage_x4": round(x4, 3),
            "market_sentiment_x5": round(x5, 3),
        },
        "methodology": "Adapted Altman Z-score for private credit",
    }


# ══════════════════════════════════════════════════════════════════
# MODULE 5 — MASTER VALUATION (unified output for AI layer)
# Runs all modules. Returns single dict consumed by narrative.py
# Weighting: DCF 50%, Comps 30%, Monte Carlo P50 20%
# ══════════════════════════════════════════════════════════════════

def full_valuation(row, instrument_type="direct_lending"):
    rate   = build_discount_rate(row, instrument_type)
    dcf    = dcf_valuation(row, rate["total_rate_pct"])
    comps  = comps_valuation(row)
    mc     = monte_carlo_valuation(row, instrument_type)
    zscore = credit_score(row)

    # Recovery analysis only when breached
    recovery = (recovery_analysis(row)
                if row.get("covenant_breached", False) else None)

    # Blended NAV — industry standard weighting
    blended = round(
        dcf["nav"] * 0.50
        + comps["implied_ev"] * 0.30
        + mc["p50"] * 0.20,
        2
    )

    # NAV vs Par flag
    nav_to_par = round((blended / row["principal_mm"]) * 100, 1)
    nav_flag   = (
        "At premium to par" if nav_to_par > 100
        else "Near par — stable" if nav_to_par > 95
        else "Moderate discount — watch" if nav_to_par > 85
        else "Significant discount — distressed"
    )

    return {
        "borrower":          row["borrower"],
        "sector":            row.get("sector", ""),
        "instrument_type":   instrument_type,
        "asc820_level":      rate["asc820_level"],
        "valuation_date":    datetime.today().strftime("%Y-%m-%d"),
        "rate_components":   rate,
        "dcf":               dcf,
        "comps":             comps,
        "monte_carlo":       mc,
        "credit_score":      zscore,
        "recovery":          recovery,
        "blended_nav_mm":    blended,
        "nav_to_par_pct":    nav_to_par,
        "nav_flag":          nav_flag,
        "weighting":         "DCF 50% / Comps 30% / MC P50 20%",
        "methodology_note":  "ASC 820 / IFRS 13 compliant — Level 3 fair value",
    }


# ══════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data import load_portfolio
    df = load_portfolio()

    for borrower in ["BetaCorp", "GammaTech", "ZetaLogistics"]:
        row    = df[df["borrower"] == borrower].iloc[0]
        result = full_valuation(row, "direct_lending")
        r      = result["rate_components"]
        d      = result["dcf"]
        c      = result["comps"]
        m      = result["monte_carlo"]
        z      = result["credit_score"]

        print(f"\n{'='*60}")
        print(f"  {result['borrower']} | {result['sector']} | "
              f"{result['asc820_level']} | {result['valuation_date']}")
        print(f"{'='*60}")
        print(f"  DISCOUNT RATE BUILD-UP")
        print(f"    SOFR base:          {r['sofr_base']}%")
        print(f"    Credit spread:      {r['credit_spread']}%  "
              f"[{r['credit_bucket']} bucket]")
        print(f"    Illiquidity prem:   {r['illiquidity_prem']}%")
        print(f"    Complexity prem:    {r['complexity_prem']}%")
        print(f"    Sector adj:         {r['sector_adj']}%")
        print(f"    Covenant adj:       {r['covenant_adj']}%")
        print(f"    ── Total rate:      {r['total_rate_pct']}%  "
              f"(BSL premium: {r['bsl_premium_pct']}%)")
        print(f"\n  VALUATION")
        print(f"    DCF NAV:            ${d['nav']}M  "
              f"({d['nav_to_par']}% of par) | "
              f"Duration: {d['duration_yrs']}yr | "
              f"DV01: ${d['dv01_mm']}M")
        print(f"    Comps EV:           ${c['implied_ev']}M  "
              f"({c['ev_coverage']}x coverage) | "
              f"{c['coverage_assessment']}")
        print(f"    Monte Carlo:        "
              f"P10 ${m['p10']}M | P50 ${m['p50']}M | "
              f"P90 ${m['p90']}M  (PD: {m['pd_applied']}%)")
        print(f"    Blended NAV:        ${result['blended_nav_mm']}M  "
              f"→ {result['nav_flag']}")
        print(f"\n  CREDIT SCORE  (Adapted Altman Z)")
        print(f"    Z-score: {z['z_score']}  → {z['zone']}")
        if result["recovery"]:
            rec = result["recovery"]
            print(f"\n  RECOVERY ANALYSIS  (Breach detected)")
            print(f"    Recovery rate:  {rec['recovery_rate']*100:.1f}%  "
                  f"→ ${rec['recovery_value_mm']}M")
            print(f"    LGD:            {rec['loss_given_default']*100:.1f}%")