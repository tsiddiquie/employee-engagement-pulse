import os
import time
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------
# Slack SDK
# -------------------------------
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# -------------------------------
# Hume AI SDK (with graceful fallback)
# -------------------------------
HUME_AVAILABLE = True
try:
    from hume import HumeClient
    try:
        from hume.core.api_error import ApiError
    except Exception:
        class ApiError(Exception):
            pass
except Exception:
    HUME_AVAILABLE = False
    class HumeClient:  # type: ignore
        def __init__(self, api_key: str): ...
    class ApiError(Exception):
        pass

# -------------------------------
# OpenAI (optional) for insights
# -------------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ===============================
# Page / Secrets / Defaults
# ===============================
st.set_page_config(page_title="Employee Engagement Pulse", page_icon="🫶", layout="wide")
st.title("🫶 Employee Engagement Pulse")
st.caption("Slack-integrated weekly emotion dashboard — Top-3 emotions per message (Hume AI), daily→weekly trends, burnout warnings, and manager insights.")

SLACK_BOT_TOKEN   = st.secrets.get("SLACK_BOT_TOKEN") or os.getenv("SLACK_BOT_TOKEN") or ""
SLACK_CHANNELS    = [c.strip() for c in (st.secrets.get("SLACK_CHANNELS") or os.getenv("SLACK_CHANNELS") or "").split(",") if c.strip()]
HUME_API_KEY      = st.secrets.get("HUME_API_KEY") or os.getenv("HUME_API_KEY") or ""
OPENAI_API_KEY    = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
OPENAI_CHAT_MODEL = st.secrets.get("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
DEF_PRESENT = (st.secrets.get("PRESENTATION_MODE") or os.getenv("PRESENTATION_MODE") or "true").lower() in ("1","true","yes","on")


# ===============================
# Slack Helpers
# ===============================
def slack_client() -> Optional[WebClient]:
    if not SLACK_BOT_TOKEN:
        return None
    return WebClient(token=SLACK_BOT_TOKEN)

@st.cache_data(show_spinner=False, ttl=300)
def list_channels_cached() -> List[Dict[str, str]]:
    client = slack_client()
    if not client:
        return []
    chans = []
    cursor = None
    while True:
        resp = client.conversations_list(
            exclude_archived=True,
            types="public_channel,private_channel",
            limit=500,
            cursor=cursor
        )
        chans.extend([{"id": c["id"], "name": c["name"]} for c in resp["channels"]])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return chans

def _history_page(client: WebClient, channel_id: str, oldest: float, cursor: Optional[str]):
    return client.conversations_history(
        channel=channel_id,
        oldest=str(oldest),
        limit=200,
        cursor=cursor,
        include_all_metadata=True
    )

def _thread_replies(client: WebClient, channel_id: str, parent_ts: str, oldest: float):
    replies = []
    cursor = None
    while True:
        resp = client.conversations_replies(
            channel=channel_id,
            ts=parent_ts,
            oldest=str(oldest),
            limit=200,
            cursor=cursor,
            include_all_metadata=True
        )
        replies.extend(resp.get("messages", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return replies

def fetch_messages(channel_ids: List[str], oldest: float, include_threads: bool, max_messages_per_channel: int) -> List[Dict]:
    client = slack_client()
    rows: List[Dict] = []
    if not client:
        return rows
    for ch in channel_ids:
        fetched = 0
        cursor = None
        try:
            while True:
                page = _history_page(client, ch, oldest, cursor)
                msgs = page.get("messages", [])
                for m in msgs:
                    rows.append({
                        "channel": ch,
                        "ts": float(m.get("ts", 0) or 0),
                        "user": m.get("user") or m.get("bot_id") or "unknown",
                        "text": m.get("text", "") or "",
                        "reactions": m.get("reactions", []),
                        "thread_ts": m.get("thread_ts", m.get("ts")),
                        "is_thread_reply": False,
                        "parent_ts": None
                    })
                    fetched += 1

                    if include_threads and m.get("reply_count", 0) > 0 and m.get("ts"):
                        try:
                            for r in _thread_replies(client, ch, m["ts"], oldest):
                                if r.get("ts") == m["ts"]:
                                    continue
                                rows.append({
                                    "channel": ch,
                                    "ts": float(r.get("ts", 0) or 0),
                                    "user": r.get("user") or r.get("bot_id") or "unknown",
                                    "text": r.get("text", "") or "",
                                    "reactions": r.get("reactions", []),
                                    "thread_ts": r.get("thread_ts", m["ts"]),
                                    "is_thread_reply": True,
                                    "parent_ts": m["ts"]
                                })
                                fetched += 1
                        except SlackApiError:
                            pass

                    if fetched >= max_messages_per_channel:
                        break
                if fetched >= max_messages_per_channel:
                    break
                cursor = page.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
        except SlackApiError as e:
            rows.append({
                "channel": ch, "ts": 0.0, "user": "error",
                "text": f"SLACK_ERROR: {e}", "reactions": [],
                "thread_ts": None, "is_thread_reply": False, "parent_ts": None
            })
    return rows


# ===============================
# DEMO DATA (Build/Test mode)
# ===============================
POSITIVE = [
    "Great momentum this week! Shipped the feature 🎉🚀",
    "Huge thanks team — proud of the collaboration here.",
    "Love the initiative on onboarding revamp!",
    "That fix was elegant and fast. Nicely done.",
    "Feeling pumped for the next sprint."
]
NEUTRAL = [
    "Sync moved to 10am; doc updated.",
    "Heads up: vendor call rescheduled.",
    "I'll take the next ticket in the queue.",
    "Noted the comments; will revise.",
    "Pushing a patch shortly."
]
NEGATIVE = [
    "I'm worried about timelines; lots of blockers.",
    "This rollout has been stressful and exhausting.",
    "Feeling confused about priorities right now.",
    "There's mounting frustration around env failures.",
    "I'm tired — context switching is burning me out."
]
DEMO_CHANNELS = ["DEMO-eng", "DEMO-product", "DEMO-general"]

def generate_demo_rows(n: int = 48, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    now = datetime.now(timezone.utc)
    rows: List[Dict] = []
    all_msgs = POSITIVE + NEUTRAL + NEGATIVE
    for _ in range(n):
        days_back = random.randint(0, 9)
        hour = random.choice([9, 11, 13, 15, 17, 19, 21])
        rows.append({
            "channel": random.choice(DEMO_CHANNELS),
            "ts": (now - timedelta(days=days_back)).replace(hour=hour, minute=0, second=0, microsecond=0).timestamp(),
            "user": random.choice(["Aisha","Ben","Chris","Divya","Eli"]),
            "text": random.choice(all_msgs),
            "reactions": [],
            "thread_ts": None,
            "is_thread_reply": False,
            "parent_ts": None
        })
    return rows


# ===============================
# Hume AI Integration
# ===============================
NEGATIVE_SET = {
    "Anger","Anxiety","Distress","Sadness","Tiredness","Disappointment",
    "Disapproval","Contempt","Guilt","Shame","Fear","Embarrassment","Horror","Pain","Grief"
}

def fallback_top3(texts: List[str]) -> List[List[Tuple[str, float]]]:
    out = []
    for t in texts:
        tl = (t or "").lower()
        if any(w in tl for w in ["great","thanks","love","proud","awesome","celebrate","🎉","🚀","👏","👍","😊","😍"]):
            out.append([("Joy",0.65),("Admiration",0.55),("Gratitude",0.45)])
        elif any(w in tl for w in ["worried","confused","tired","exhaust","stress","frustrat","burn","deadline","on-call","oncall","weekend","weary","😩"]):
            out.append([("Anxiety",0.62),("Tiredness",0.50),("Confusion",0.48)])
        else:
            out.append([("Calmness",0.40),("Interest",0.38),("Concentration",0.35)])
    return out

def extract_top3_from_predictions(preds_obj, expected: int) -> List[List[Tuple[str, float]]]:
    result: List[List[Tuple[str, float]]] = []
    sources = preds_obj if isinstance(preds_obj, list) else preds_obj.get("predictions") or preds_obj
    if not isinstance(sources, list):
        return result
    for source in sources:
        inner_preds = source.get("results", {}).get("predictions", []) if isinstance(source, dict) else []
        for item in inner_preds:
            lang = item.get("models", {}).get("language", {})
            gps = lang.get("grouped_predictions", [])
            score_map: Dict[str, float] = {}
            for gp in gps:
                for p in gp.get("predictions", []):
                    for emo in p.get("emotions", []):
                        name = emo.get("name")
                        if not name:
                            continue
                        score = float(emo.get("score", 0.0))
                        score_map[name] = max(score_map.get(name, 0.0), score)
            if not score_map:
                result.append([("Calmness",0.35),("Interest",0.33),("Concentration",0.30)])
            else:
                top3 = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
                result.append([(n, s) for n, s in top3])
    if len(result) < expected:
        result.extend([[("Calmness",0.35),("Interest",0.33),("Concentration",0.30)]]*(expected-len(result)))
    return result[:expected]

def hume_top3_emotions(texts: List[str], api_key: Optional[str]) -> List[List[Tuple[str, float]]]:
    if not HUME_AVAILABLE or not api_key:
        return fallback_top3(texts)

    client = HumeClient(api_key=api_key)

    def _start(gran: str):
        return client.expression_measurement.batch.start_inference_job(
            models={"language": {"granularity": gran}},
            text=texts,
            notify=False
        )

    for gran in ("sentence", "word"):
        try:
            start_res = _start(gran)
            break
        except ApiError as e:
            if "Granularity" in str(e) or "granularity" in str(e).lower():
                continue
            return fallback_top3(texts)
    else:
        return fallback_top3(texts)

    job_id = getattr(start_res, "job_id", None) or (start_res.get("job_id") if isinstance(start_res, dict) else None)
    if not job_id:
        return fallback_top3(texts)

    t0 = time.time()
    while time.time() - t0 < 60:
        details = client.expression_measurement.batch.get_job_details(id=job_id)
        status = getattr(getattr(details, "state", None), "status", None)
        if status is None and isinstance(details, dict):
            status = details.get("state", {}).get("status")
        if status == "COMPLETED":
            break
        if status in {"FAILED","CANCELLED","ERROR"}:
            return fallback_top3(texts)
        time.sleep(2.0)
    else:
        return fallback_top3(texts)

    preds = client.expression_measurement.batch.get_job_predictions(id=job_id)
    if hasattr(preds, "to_dict"):
        preds = preds.to_dict()
    parsed = extract_top3_from_predictions(preds, expected=len(texts))
    return parsed or fallback_top3(texts)


# ===============================
# Aggregation / Insights
# ===============================
def compute_aggregations(df: pd.DataFrame) -> Dict:
    df["ts_dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("UTC")
    df["date"] = df["ts_dt"].dt.date
    daily = df.groupby(["date","primary_emotion"], as_index=False).size().rename(columns={"size":"count"})
    df["week"] = pd.to_datetime(df["ts_dt"]).dt.to_period("W").apply(lambda p: p.start_time.date())
    weekly = df.groupby(["week","primary_emotion"], as_index=False).size().rename(columns={"size":"count"})
    total_per_day = daily.groupby("date")["count"].sum().rename("total")
    neg = daily[daily["primary_emotion"].isin(NEGATIVE_SET)].groupby("date")["count"].sum().rename("neg")
    risk = pd.concat([total_per_day, neg], axis=1).fillna(0.0)
    risk["neg_share"] = risk["neg"] / risk["total"].replace(0,1)
    flags = []
    if (risk["neg_share"] >= 0.45).sum() >= 3:
        flags.append("⚠️ Sustained negative tone (≥45%) on 3+ days.")
    if (risk["neg_share"] > 0.40).rolling(3).sum().max() >= 3:
        flags.append("⚠️ 3-day streak of elevated anxiety/sadness.")
    ch_tbl = (df.groupby("channel", as_index=False)
                .agg(messages=("text","count"),
                     top_emotion=("primary_emotion", lambda s: s.value_counts().idxmax())))
    return {"daily": daily, "weekly": weekly, "risk": risk, "flags": flags, "channels": ch_tbl}

def actionable_insights(df: pd.DataFrame) -> List[str]:
    frac = df["primary_emotion"].value_counts(normalize=True)
    tips = []
    if frac.get("Anxiety",0)+frac.get("Tiredness",0) > 0.35:
        tips.append("Rebalance workload & protect focus time; cut low-priority work.")
    if frac.get("Confusion",0) > 0.18:
        tips.append("Clarify goals and success criteria; post a crisp weekly plan.")
    if frac.get("Anger",0)+frac.get("Disapproval",0) > 0.18:
        tips.append("Unblock build/test issues; run an incident review for chronic blockers.")
    if not tips:
        tips.append("Team tone looks healthy — amplify wins and recognition.")
    return tips[:4]

def llm_insights(summary: Dict) -> Optional[str]:
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    client = OpenAI(api_key=OPENAI_API_KEY)
    sys = ("You analyze Slack team chatter for managers. Return: "
           "1) a concise 3–4 sentence weekly summary, "
           "2) 3 concrete, action-oriented insights, "
           "3) 2–3 suggested interventions.")
    user = f"METRICS JSON:\n{summary}"
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0.4,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()


# ===============================
# Sidebar — Mode & Settings (Data source always visible)
# ===============================
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Mode",
        ["Presentation (Slack only)", "Build/Test (Demo + Sandbox)"],
        index=0 if DEF_PRESENT else 1
    )

    ds_options = ["Demo (no Slack)", "Slack"]
    if mode == "Presentation (Slack only)":
        data_source = st.radio("Data source", ds_options, index=1, disabled=True,
                               help="Switch to Build/Test to use Demo data.")
    else:
        data_source = st.radio("Data source", ds_options, index=0)

    lookback_days = st.slider("Lookback (days)", 7, 30, 14)
    include_threads = st.checkbox("Include thread replies", value=True)
    max_per_channel = st.slider("Max messages per channel", 200, 2000, 800, step=100)

    st.markdown("---")
    if HUME_API_KEY:
        st.success("Engine: Hume AI (live)", icon="✅")
    else:
        st.warning("Engine: Heuristic fallback (no HUME_API_KEY)", icon="⚠️")
    if OPENAI_API_KEY:
        st.success("OpenAI insights enabled", icon="✅")
    else:
        st.info("OpenAI insights disabled (using heuristics).", icon="ℹ️")

    st.markdown("---")
    if st.button("🧹 Clear session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ===============================
# Channel Picker (ONLY when data_source == Slack)
# ===============================
channels_selected: List[str] = []
ch_name_map: Dict[str, str] = {}
if data_source == "Slack":
    if SLACK_BOT_TOKEN:
        try:
            chans = list_channels_cached()
            ch_name_map = {c["id"]: f"#{c['name']}" for c in chans}
            pretty = {f"#{c['name']} ({c['id']})": c["id"] for c in chans}
            st.subheader("Choose Slack Channels")
            chosen = st.multiselect("Bot must be a member of each channel:", options=list(pretty.keys()))
            channels_selected = [pretty[x] for x in chosen]
        except Exception as e:
            st.error(f"Failed to list channels: {e}")
    else:
        st.error("Missing Slack token. Add SLACK_BOT_TOKEN in .streamlit/secrets.toml", icon="🚫")

    manual_ids = st.text_input("Or paste channel IDs (comma-separated)", value=",".join(SLACK_CHANNELS))
    if manual_ids.strip():
        channels_selected = list({*channels_selected, *[c.strip() for c in manual_ids.split(",") if c.strip()]})
    st.write(f"**Selected channels:** {len(channels_selected)}")
else:
    st.info("Using **Demo data** — no Slack connection required.", icon="ℹ️")


# ===============================
# Demo Controls (visible only when Demo is selected)
# ===============================
if data_source == "Demo (no Slack)":
    st.subheader("Demo Controls")
    demo_n = st.slider("How many demo messages?", 12, 200, 48, step=6)
    demo_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    demo_btn = st.button("▶️ Generate Demo & Analyze")
else:
    demo_n, demo_seed, demo_btn = None, None, False


# ===============================
# Fetch / Generate, Label, Aggregate
# ===============================
oldest = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()

if data_source == "Slack":
    go_label = "🔄 Fetch & Analyze (Slack)"
    go_disabled = (not SLACK_BOT_TOKEN or not channels_selected)
    fetch_btn = st.button(go_label, disabled=go_disabled)
else:
    fetch_btn = False

if fetch_btn or demo_btn:
    with st.spinner("Analyzing emotions…"):
        if data_source == "Slack":
            rows = fetch_messages(channels_selected, oldest, include_threads, max_messages_per_channel)
        else:
            rows = generate_demo_rows(n=demo_n or 48, seed=demo_seed or 42)

        df = pd.DataFrame(rows)
        if df.empty:
            st.warning("No messages retrieved for this window.", icon="ℹ️")
        else:
            df = df[df["ts"] > 0].copy()
            texts = df["text"].fillna("").astype(str).tolist()
            top3 = hume_top3_emotions(texts, HUME_API_KEY)
            df["top3"] = top3
            df["primary_emotion"] = [t[0][0] if t else "Calmness" for t in top3]
            aggs = compute_aggregations(df)
            st.session_state["df"] = df
            st.session_state["aggs"] = aggs
            st.session_state["summary"] = {
                "messages": int(df.shape[0]),
                "channels": len(set(df["channel"])),
                "date_min": str(pd.to_datetime(df["ts"], unit="s").dt.date.min()),
                "date_max": str(pd.to_datetime(df["ts"], unit="s").dt.date.max()),
                "top_emotions": df["primary_emotion"].value_counts().head(5).to_dict(),
                "neg_ratio": float((df["primary_emotion"].isin(list(NEGATIVE_SET))).mean()),
                "source": data_source,
            }


# ===============================
# Render Dashboard & (optional) Sandbox
# ===============================
df = st.session_state.get("df")
aggs = st.session_state.get("aggs")
summary = st.session_state.get("summary")

if df is None:
    if data_source == "Slack":
        st.info("Select channels and click **Fetch & Analyze** to build the Slack dashboard.", icon="ℹ️")
    else:
        st.info("Use the **Demo Controls** above and click **Generate Demo & Analyze**.", icon="ℹ️")
else:
    st.subheader("Weekly Overview")
    neg_ratio = summary["neg_ratio"]
    flags = aggs["flags"]
    if neg_ratio > 0.40 or flags:
        risk_label = "High" if neg_ratio > 0.45 or len(flags) >= 2 else "Medium"
    else:
        risk_label = "Low"

    k1, k2, k3 = st.columns(3)
    k1.metric("Messages analyzed", f"{summary['messages']}")
    k2.metric("Channels", f"{summary['channels']}")
    k3.metric("Burnout risk", risk_label)

    if risk_label == "High":
        st.error("🔥 Burnout warning: **High risk** this period.", icon="🔥")
    elif risk_label == "Medium":
        st.warning("⚠️ Burnout warning: **Medium risk**.", icon="⚠️")
    else:
        st.success("✅ Burnout risk: **Low**.", icon="✅")

    for f in flags:
        st.warning(f)

    st.subheader("Daily Emotion Mix")
    daily = aggs["daily"].copy()
    fig = px.bar(daily, x="date", y="count", color="primary_emotion", barmode="stack", height=360)
    fig.update_layout(xaxis_title="Date", yaxis_title="Messages")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekly Mix (Share of Messages)")
    weekly = aggs["weekly"].copy()
    weekly["total"] = weekly.groupby("week")["count"].transform("sum")
    weekly["share"] = weekly["count"] / weekly["total"].replace(0,1)
    fig2 = px.bar(weekly, x="week", y="share", color="primary_emotion", barmode="stack", height=320)
    fig2.update_layout(xaxis_title="Week starting", yaxis_title="Share")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Channels — Volume & Top Emotion")
    ch_tbl = aggs["channels"].copy()
    st.dataframe(ch_tbl, use_container_width=True)

    st.subheader("Actionable Insights for Managers")
    llm = llm_insights({
        "messages": summary["messages"],
        "channels": summary["channels"],
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "top_emotions": summary["top_emotions"],
        "neg_ratio": summary["neg_ratio"],
        "source": summary["source"],
    })
    if llm:
        st.markdown(llm)
    else:
        for tip in actionable_insights(df):
            st.markdown(f"- {tip}")

    st.subheader("Messages (labeled)")
    show = df.copy()
    show["ts_dt"] = pd.to_datetime(show["ts"], unit="s", utc=True).dt.tz_convert("UTC")
    show["Top 3 emotions"] = [", ".join([n for (n, _s) in t]) for t in show["top3"]]
    st.dataframe(show[["ts_dt","channel","user","text","primary_emotion","Top 3 emotions","is_thread_reply","parent_ts"]],
                 use_container_width=True, height=380)
    st.download_button("⬇️ Download CSV", data=show.to_csv(index=False).encode("utf-8"),
                       file_name=f"pulse_messages_{summary['source'].lower()}.csv", mime="text/csv")

# Sandbox only in Build/Test mode
if mode == "Build/Test (Demo + Sandbox)":
    st.markdown("---")
    st.subheader("🧪 Emotion Sandbox")
    text = st.text_area("Type a sample message", value="We crushed the launch! 🎉 Thanks team! 🚀", height=80)
    if st.button("Analyze message"):
        engine = "Hume" if HUME_API_KEY else "Heuristic"
        with st.spinner(f"Analyzing with {engine} engine…"):
            top3 = hume_top3_emotions([text], HUME_API_KEY)[0]
        st.markdown("**Top-3 emotions:** " + ", ".join([f"{n}" for (n, _s) in top3]))
        st.caption("Labels reflect *expressed* tone in text (Hume AI or local heuristic fallback).")

st.markdown("---")
st.caption("Privacy: Slack messages are read via your bot token; emotion labels are computed in memory. Demo mode uses synthetic messages. No data is persisted server-side.")