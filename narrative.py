from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ── CLIENT INITIALIZATION ────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not set")

client = Groq(api_key=api_key)


def generate_credit_narrative(row, valuation_result=None):
    """Generate structured credit memo"""

    base_data = f"""
Borrower: {row['borrower']}
Sector: {row['sector']}
Principal: ${row['principal_mm']}M
Leverage: {row['net_leverage']}x
Interest Coverage: {row['interest_coverage']}x
LTV: {row['ltv_pct']}%
Sentiment: {row['sentiment_score']}
Risk Flags: {int(row['risk_flags'])}
"""

    val_data = ""
    if valuation_result:
        val_data = f"""
Blended NAV: ${valuation_result['blended_nav_mm']}M
Discount Rate: {valuation_result['rate_components']['total_rate_pct']}%
Z-Score: {valuation_result['credit_score']['z_score']}
"""

    prompt = f"""
You are a senior private credit analyst.

Write a sharp, professional credit memo with 4 sections:

HEALTH SUMMARY  
VALUATION ASSESSMENT  
KEY RISKS  
RECOMMENDED ACTION  

Use numbers. Be decisive. No fluff.

{base_data}
{val_data}
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a credit analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating narrative: {str(e)}"