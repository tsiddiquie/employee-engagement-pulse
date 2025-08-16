import time
from typing import Dict, List, Any, Tuple, Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def get_client(token: str) -> WebClient:
    return WebClient(token=token)

def list_channels(token: str) -> List[Dict[str, str]]:
    """Return visible channels (public + private where bot is a member)."""
    client = get_client(token)
    channels = []
    cursor = None
    while True:
        resp = client.conversations_list(
            exclude_archived=True,
            types="public_channel,private_channel",
            limit=500,
            cursor=cursor
        )
        channels.extend([{"id": c["id"], "name": c["name"]} for c in resp["channels"]])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return channels

def _fetch_history_page(client: WebClient, channel_id: str, oldest: float, cursor: Optional[str]) -> Dict[str, Any]:
    return client.conversations_history(
        channel=channel_id,
        oldest=str(oldest),
        limit=200,
        cursor=cursor,
        include_all_metadata=True
    )

def _fetch_replies(client: WebClient, channel_id: str, parent_ts: str, oldest: float) -> List[Dict[str, Any]]:
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

def fetch_messages(
    token: str,
    channel_ids: List[str],
    oldest: float,
    include_threads: bool = True,
    max_messages_per_channel: int = 1000,
    sleep_sec: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Fetch messages for channels since `oldest` (epoch seconds).
    Includes reactions and (optionally) threaded replies.
    """
    client = get_client(token)
    all_rows: List[Dict[str, Any]] = []

    for ch in channel_ids:
        try:
            fetched = 0
            cursor = None
            while True:
                page = _fetch_history_page(client, ch, oldest, cursor)
                msgs = page.get("messages", [])
                for m in msgs:
                    row = {
                        "channel": ch,
                        "ts": float(m.get("ts", 0)),
                        "user": m.get("user") or m.get("bot_id") or "unknown",
                        "text": m.get("text", "") or "",
                        "reactions": m.get("reactions", []),
                        "thread_ts": m.get("thread_ts", m.get("ts")),
                        "is_thread_reply": False,
                        "parent_ts": None,
                    }
                    all_rows.append(row)
                    fetched += 1

                    # Fetch thread replies
                    if include_threads and m.get("reply_count", 0) > 0 and m.get("ts"):
                        try:
                            replies = _fetch_replies(client, ch, m["ts"], oldest)
                            for r in replies:
                                if r.get("ts") == m["ts"]:
                                    continue  # skip parent duplication
                                all_rows.append({
                                    "channel": ch,
                                    "ts": float(r.get("ts", 0)),
                                    "user": r.get("user") or r.get("bot_id") or "unknown",
                                    "text": r.get("text", "") or "",
                                    "reactions": r.get("reactions", []),
                                    "thread_ts": r.get("thread_ts", m["ts"]),
                                    "is_thread_reply": True,
                                    "parent_ts": m["ts"],
                                })
                                fetched += 1
                        except SlackApiError:
                            # ignore thread errors; continue
                            pass

                    if fetched >= max_messages_per_channel:
                        break

                if fetched >= max_messages_per_channel:
                    break
                cursor = page.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
                time.sleep(sleep_sec)

        except SlackApiError as e:
            # Collect an error row to surface in UI
            all_rows.append({
                "channel": ch, "ts": 0.0, "user": "error", "text": f"SLACK_ERROR: {e}", "reactions": [],
                "thread_ts": None, "is_thread_reply": False, "parent_ts": None
            })
    return all_rows