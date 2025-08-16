from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from emoji import demojize
import re

analyzer = SentimentIntensityAnalyzer()

# Minimal emoji & reaction sentiment map [-2..+2]
EMOJI_SENTIMENT: Dict[str, int] = {
    # positive
    "smile": 2, "grin": 2, "blush": 2, "heart_eyes": 2, "tada": 2, "party_popper": 2, "sparkles": 2,
    "white_check_mark": 1, "thumbsup": 1, "+1": 1, "clap": 1, "pray": 1, "rocket": 2,
    # neutral-ish
    "thinking_face": 0, "eyes": 0, "neutral_face": 0,
    # negative
    "thumbsdown": -1, "-1": -1, "cry": -2, "sob": -2, "weary": -2, "tired_face": -2,
    "rage": -2, "angry": -2, "skull": -2, "fire": -1, "warning": -1, "face_with_steam_from_nose": -1,
}

BURNOUT_KEYWORDS = [
    "burnout","burn out","exhausted","exhausting","overtime","over time","weekend work",
    "crunch","deadline","late night","late-night","oncall","on-call","pager","overwhelmed",
    "pressure","stressed","stressful","fatigue","tired","stretching","capacity","no bandwidth"
]

STOPWORDS = set("""
the a an and or but if to in of for on at from by with about into against between without within
is are was were be been being do does did have has had having it this that these those i you we they
as not can will would should could our your their my me us them he she his her its
""".split())

def text_sentiment(text: str) -> float:
    """VADER compound [-1,1]."""
    return float(analyzer.polarity_scores(text or "")["compound"])

def emoji_sent_from_text(text: str) -> float:
    """Parse unicode emojis via demojize(':name:') and average mapped valence."""
    if not text:
        return 0.0
    names = re.findall(r":([a-z0-9_+\-]+):", demojize(text, language="en"))
    vals = [EMOJI_SENTIMENT.get(n, 0) for n in names]
    return (sum(vals)/len(vals)) if vals else 0.0

def reactions_sentiment(reactions: List[dict]) -> float:
    """Map Slack reaction names to valence, weighted by count."""
    total = 0
    score = 0
    for r in reactions or []:
        name = r.get("name","")
        count = int(r.get("count", 0))
        val = EMOJI_SENTIMENT.get(name, 0)
        score += val * count
        total += count
    return (score/total) if total > 0 else 0.0

def combined_sentiment(text_s: float, emoji_s: float, react_s: float) -> float:
    """Blend signals into [-1,1]."""
    return 0.7*text_s + 0.15*emoji_s + 0.15*react_s

def burnout_flags(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for k in BURNOUT_KEYWORDS if k in t)

def tokenize_words(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]