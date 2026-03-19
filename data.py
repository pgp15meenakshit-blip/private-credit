import pandas as pd

def load_portfolio():
    df = pd.read_csv("portfolio.csv")
    df["covenant_breached"] = df["net_leverage"] > df["leverage_limit"]
    df["low_coverage"] = df["interest_coverage"] < 2.0
    df["high_ltv"] = df["ltv_pct"] > 75
    df["neg_sentiment"] = df["sentiment_score"] < -0.2
    flag_cols = ["covenant_breached", "low_coverage", "high_ltv", "neg_sentiment"]
    df["risk_flags"] = df[flag_cols].sum(axis=1)
    return df

if __name__ == "__main__":
    df = load_portfolio()
    print(df[["borrower", "risk_flags", "covenant_breached", "low_coverage"]].to_string())