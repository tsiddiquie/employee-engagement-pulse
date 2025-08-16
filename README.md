Employee Engagement Pulse (Streamlit)

Weekly sentiment dashboard from Slack: reads selected channels (incl. threads + reactions), labels each message with Top-3 emotions (Hume AI), aggregates daily → weekly, flags burnout risk, and generates actionable insights for managers. (Optional legacy numeric mode is still supported.)

Quick Start (Local)

Create venv and install:

python -m venv venv
venv/Scripts/activate  # Windows, or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt


Create .streamlit/secrets.toml and fill your keys:

# Slack (optional for live data)
SLACK_BOT_TOKEN = "xoxb-YOUR-TOKEN"
SLACK_CHANNELS = "C0123ABC,G0456DEF"  # optional defaults

# OpenAI (optional for Insights)
OPENAI_API_KEY = "sk-..." 
OPENAI_CHAT_MODEL = "gpt-4o-mini"     # optional (defaults in app)

# Hume AI (recommended for emotion labels)
HUME_API_KEY = "your-hume-key"

# UI default (optional): hide demo/sandbox for judging
PRESENTATION_MODE = "false"


Run:

streamlit run streamlit_app.py

Slack Bot Setup

Go to api.slack.com/apps → Create New App → From scratch

Add these OAuth Scopes (OAuth & Permissions page):

conversations:read

channels:history

groups:history

reactions:read

users:read

Install to Workspace and copy the Bot User OAuth Token (xoxb-...)

Invite the bot to channels you want to analyze: /invite @YourBotName

Features

Multi-channel analysis: public & private channels where the bot is a member

Thread support: includes replies to maintain context

Emotion labels (Hume AI): Top-3 emotions per message; primary emotion drives daily/weekly aggregates

Uses language granularity "sentence" with automatic fallback to "word"

Burnout detection: rule-based warnings from sustained negative emotions (e.g., Anxiety, Sadness, Anger, Tiredness)

AI insights: optional OpenAI summary + concrete recommendations for managers

Demo & Sandbox: test without Slack; switch modes in the left sidebar

Presentation mode: hides demo/sandbox; focuses on the Slack dashboard

Export: download raw labeled data as CSV

Privacy-first: no data stored server-side; runs locally or on Streamlit Cloud

(Optional) Legacy numeric: prior pipeline (text + emoji + reactions → score) still available if enabled

Note: reactions are fetched for context and for the legacy numeric mode; Hume labeling focuses on message text.

Example Dashboard

Weekly metrics: Messages analyzed, Burnout risk (Low/Medium/High)

(If legacy numeric mode is enabled: also avg sentiment (−1 to +1) and trend delta)

Daily trend chart: primary emotion mix by day

Weekly mix: share of top emotions week-over-week

Channel breakdown: compare volume and top emotions by channel

Manager insights: AI-generated (OpenAI) or heuristic recommendations

Deployment (Streamlit Cloud)

Fork this repo to your GitHub

Deploy via share.streamlit.io

Add secrets in the Streamlit Cloud dashboard (same format as local secrets.toml)

Deploy — your team dashboard is live!

Toggle Build/Test (Demo + Sandbox) vs Presentation in the left sidebar

Privacy & Ethics

Transparency: let your team know the tool exists and what it analyzes

Aggregate focus: for team health trends, not individual performance

Consent: align on which channels are included and how insights are shared

Data handling: the bot token grants read access—avoid highly sensitive channels

Interpretation: emotion labels reflect expressed tone in text; treat warnings as signals, not diagnoses

This tool helps managers spot trends and take proactive steps to support team wellbeing. Use insights constructively and pair them with direct team feedback for best results.