"""Streamlit dashboard for limen.

Run with:

    pip install -e ".[ui]"
    streamlit run ui/app.py
"""

from __future__ import annotations

import json

import streamlit as st

from limen.cli import build_default_bot
from limen.logger import Logger


# ---------- page setup ----------

st.set_page_config(page_title="limen", page_icon="🧠", layout="wide")
st.title("limen")
st.caption("A Global Workspace model — conscious broadcast vs. unconscious remainder.")


# ---------- session state ----------

if "bot" not in st.session_state:
    st.session_state.bot = None

if "history" not in st.session_state:
    st.session_state.history = []


# ---------- sidebar ----------

with st.sidebar:
    st.header("Settings")

    capacity = st.slider(
        "Workspace capacity",
        min_value=1, max_value=6, value=3,
        help="How many items can be broadcast per cycle.",
    )

    memory_feedback = st.toggle(
        "Memory feedback",
        value=True,
        help="If on, broadcast items feed back into the salience context.",
    )

    log_to_db = st.toggle("Log to SQLite", value=False)
    db_path = st.text_input("Log path", value="data/ui_logs.db", disabled=not log_to_db)

    if st.button("Reset bot & memory"):
        st.session_state.bot = None
        st.session_state.history = []
        st.success("Reset.")

    if st.button("Clear history only"):
        st.session_state.history = []
        st.success("History cleared.")


# ---------- bot lifecycle ----------

def _make_bot():
    logger = Logger(db_path) if log_to_db else None
    if logger is not None:
        logger.open()
    return build_default_bot(
        capacity=capacity,
        logger=logger,
        memory_feedback=memory_feedback,
    )


# rebuild bot if capacity / memory_feedback changed, or if not built yet
_sig = (capacity, memory_feedback, log_to_db, db_path)
if st.session_state.get("_sig") != _sig or st.session_state.bot is None:
    # close old logger if any
    old_bot = st.session_state.bot
    if old_bot is not None and getattr(old_bot, "logger", None) is not None:
        try:
            old_bot.logger.close()
        except Exception:
            pass

    st.session_state.bot = _make_bot()
    st.session_state["_sig"] = _sig
    # rebuilding drops memory history in the bot; keep UI history as-is


# ---------- input ----------

text = st.text_area(
    "Input text",
    value="I love this, what is it?",
    height=100,
)

col_run, col_json = st.columns([1, 3])
run = col_run.button("Run", type="primary")
show_json = col_json.toggle("Show raw JSON", value=False)


# ---------- run cycle ----------

if run:
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        cycle = st.session_state.bot.respond(text)
        described = st.session_state.bot.describe(cycle)
        st.session_state.history.insert(0, described)
        st.session_state.history = st.session_state.history[:20]


# ---------- render latest cycle ----------

def _render_item(item: dict, dim: bool = False) -> None:
    src = item.get("source", "?")
    sal = item.get("salience", 0.0)
    payload = item.get("payload", {})

    header = f"**{src}** — salience `{sal:.3f}`"
    if dim:
        header = f"<span style='color:#888'>{header}</span>"
        st.markdown(header, unsafe_allow_html=True)
    else:
        st.markdown(header)

    pretty = ", ".join(f"{k}={v!r}" for k, v in payload.items())
    if dim:
        st.markdown(f"<span style='color:#888;font-size:0.9em'>{pretty}</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(pretty)


if st.session_state.history:
    latest = st.session_state.history[0]

    st.subheader(f"Cycle {latest['cycle_id']} — capacity {latest['capacity']}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Conscious (broadcast)")
        if latest["broadcast"]:
            for item in latest["broadcast"]:
                with st.container(border=True):
                    _render_item(item, dim=False)
        else:
            st.info("Nothing entered consciousness.")

    with right:
        st.markdown("### Unconscious")
        if latest["unconscious"]:
            for item in latest["unconscious"]:
                with st.container(border=True):
                    _render_item(item, dim=True)
        else:
            st.info("All items were broadcast.")

    if latest["module_errors"]:
        st.error("Module errors:")
        for e in latest["module_errors"]:
            st.write(f"- `{e['module']}`: {e['error']}")

    if show_json:
        st.code(json.dumps(latest, indent=2), language="json")


# ---------- history panel ----------

if len(st.session_state.history) > 1:
    with st.expander(f"History ({len(st.session_state.history)} cycles)"):
        for h in st.session_state.history[1:]:
            b = ", ".join(i["source"] for i in h["broadcast"]) or "—"
            u = ", ".join(i["source"] for i in h["unconscious"]) or "—"
            st.markdown(f"**{h['cycle_id']}** — conscious: `{b}` · unconscious: `{u}`")
