from typing import Dict, Any, List, Tuple
from collections import Counter
from datetime import datetime, timedelta
import os

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from sentiment import tokenize_words

def _fallback_insights(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic insights when no OpenAI key: top topics, positives, negatives, suggestions.
    """
    tokens_pos = summary.get("tokens_pos", [])
    tokens_neg = summary.get("tokens_neg", [])
    common_pos = [w for w,_ in Counter(tokens_pos).most_common(5)]
    common_neg = [w for w,_ in Counter(tokens_neg).most_common(5)]

    suggestions = []
    if summary.get("burnout_risk") in ("Medium","High"):
        suggestions.append("Address workload signals; consider rotating on-call and clarifying priorities.")
    if summary.get("trend_delta",0) < -0.1:
        suggestions.append("Sentiment trending down; run a listening session and acknowledge top concerns.")
    if not suggestions:
        suggestions.append("Celebrate recent wins and reinforce norms for constructive feedback.")

    return {
        "summary": f"Weekly avg sentiment {summary.get('weekly_avg'):.2f} with trend delta {summary.get('trend_delta'):.2f}. "
                   f"Top positives: {', '.join(common_pos) or '—'}. Top negatives: {', '.join(common_neg) or '—'}.",
        "insights": [
            "Maintain visible recognition for helpful contributions.",
            "Clarify timelines where deadlines are causing stress.",
            "Encourage breaks; nudge managers to model healthy hours.",
        ],
        "suggestions": suggestions
    }

def generate_insights(summary: Dict[str, Any], openai_api_key: str = "", model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Use OpenAI if available, otherwise fallback.
    """
    if not openai_api_key or OpenAI is None:
        return _fallback_insights(summary)

    client = OpenAI(api_key=openai_api_key)
    sys = (
        "You analyze Slack team chatter for managers. "
        "Given the metrics and top tokens, produce:\n"
        "1) A crisp 3–4 sentence weekly summary.\n"
        "2) 3 concrete, action-oriented insights.\n"
        "3) 2–3 suggested interventions, sensitive and constructive.\n"
        "Keep it practical and non-judgmental."
    )
    user = f"""
METRICS JSON:
{summary}
"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,
        messages=[
            {"role":"system","content":sys},
            {"role":"user","content":user}
        ]
    )
    content = resp.choices[0].message.content.strip()
    return {"llm_text": content}