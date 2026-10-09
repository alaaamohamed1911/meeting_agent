import os
import re
import threading
import uuid
from datetime import date

import streamlit as st
from langgraph.types import Command

st.set_page_config(page_title="Meeting → Action Agent", page_icon="📋", layout="wide")

# Configure the Gemini API key in Streamlit Cloud Secrets, never in GitHub.
try:
    configured_secrets = dict(st.secrets)
except Exception:
    configured_secrets = {}
for key in ("GEMINI_API_KEY", "MEETING_AGENT_STORAGE"):
    if configured_secrets.get(key) and not os.environ.get(key):
        os.environ[key] = str(configured_secrets[key])

st.title("📋 Meeting → Action Agent")
st.caption("Gemini + LangGraph · Tool selection · Human clarification · Approval · SQLite")

# Import after configuring secret-driven environment variables.
from agent import (
    create_app_graph, get_pending_interrupt, new_config,
    initial_state, get_tasks, MAX_TRANSCRIPT_CHARS,
)

@st.cache_resource
def resources():
    graph, conn = create_app_graph()
    return graph, conn, threading.RLock()

graph, checkpoint_connection, graph_lock = resources()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None


def pending(thread_id):
    with graph_lock:
        config = new_config(thread_id)
        request = get_pending_interrupt(graph, config)
        state = graph.get_state(config).values
    return request, state


def resume(payload):
    with graph_lock:
        graph.invoke(Command(resume=payload), config=new_config(st.session_state.thread_id))
    st.rerun()


with st.sidebar:
    st.subheader("Workflow")
    if st.button("New meeting", use_container_width=True):
        st.session_state.thread_id = None
        st.rerun()
    st.caption("SQLite files must live on persistent storage for durable data and paused workflows.")
    st.caption("This app is designed for a single trusted reviewer, not a multi-tenant production service.")
    with st.expander("Resume paused workflow"):
        restore = st.text_input("Thread ID", placeholder="Paste your workflow ID")
        if st.button("Open workflow"):
            if not re.fullmatch(r"[a-fA-F0-9-]{36}", restore.strip()):
                st.error("Enter a valid UUID")
            else:
                st.session_state.thread_id = restore.strip()
                st.rerun()

if not st.session_state.thread_id:
    with st.form("new_meeting", clear_on_submit=False):
        meeting_title = st.text_input("Meeting title")
        meeting_date = st.date_input("Meeting date", value=date.today())
        transcript = st.text_area("Transcript (Arabic or English)", height=270)
        submitted = st.form_submit_button("Analyze meeting", type="primary")
    if submitted:
        if not meeting_title.strip() or not transcript.strip():
            st.error("Meeting title and transcript are required")
        elif len(transcript) > MAX_TRANSCRIPT_CHARS:
            st.error(f"Transcript must be at most {MAX_TRANSCRIPT_CHARS:,} characters")
        else:
            thread_id = str(uuid.uuid4())
            try:
                with st.spinner("Analyzing meeting…"):
                    with graph_lock:
                        graph.invoke(
                            initial_state(meeting_title.strip(), meeting_date.isoformat(), transcript),
                            config=new_config(thread_id),
                        )
                st.session_state.thread_id = thread_id
                st.rerun()
            except Exception as exc:
                st.error(f"Analysis failed: {type(exc).__name__}. Check configuration or logs.")
else:
    thread_id = st.session_state.thread_id
    st.code(thread_id, language=None)
    try:
        request, state = pending(thread_id)
    except Exception as exc:
        st.error(f"Unable to load this workflow: {type(exc).__name__}")
        st.stop()

    if not state:
        st.warning("Workflow not found. Check the ID and persistent checkpoint storage.")
        st.stop()

    analysis = state.get("analysis") or {}
    if analysis:
        st.subheader("Meeting summary")
        st.write(analysis.get("summary", ""))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Decisions**")
            for item in analysis.get("decisions", []): st.write("•", item)
        with col2:
            st.markdown("**Suggestions (not approved)**")
            for item in analysis.get("suggestions", []): st.write("•", item)

    if request and request.get("message") == "Clarify missing task information":
        st.subheader("1 · Human clarification")
        st.info("Only supply confirmed information. Event-based deadlines can remain empty.")
        with st.form("clarifications"):
            answers = {}
            for item in request.get("requests", []):
                index = item["task_index"]
                with st.container(border=True):
                    st.markdown(f"**Task {index}: {item.get('title','')}**")
                    st.caption(item.get("question", ""))
                    original = (analysis.get("tasks") or [])[index]
                    owner = st.text_input("Owner (optional)", value=original.get("owner") or "", key=f"owner_{index}")
                    deadline = st.text_input("Deadline YYYY-MM-DD (optional)", value="", key=f"date_{index}", placeholder="Leave blank for event-based tasks")
                    change = {}
                    if owner.strip() and owner.strip() != (original.get("owner") or ""):
                        change["owner"] = owner.strip()
                    if deadline.strip():
                        change["deadline_date"] = deadline.strip()
                    if change: answers[str(index)] = change
            go = st.form_submit_button("Continue to approval", type="primary")
        if go:
            bad = []
            for index, update in answers.items():
                if update.get("deadline_date"):
                    try: date.fromisoformat(update["deadline_date"])
                    except ValueError: bad.append(index)
            if bad:
                st.error("Invalid deadline date for task(s): " + ", ".join(bad))
            else:
                resume(answers)

    elif request and request.get("message") == "Review each task":
        st.subheader("2 · Human approval")
        st.warning("Tasks marked needs_review may lack an owner or fixed deadline. Approve only if intentional.")
        with st.form("approvals"):
            choices = {}
            for item in request["tasks"]:
                index = str(item["task_id"])
                with st.container(border=True):
                    st.markdown(f"**{item['title']}**")
                    st.write(f"Owner: {item.get('owner') or 'Not assigned'}")
                    st.write(f"Deadline: {item.get('deadline') or 'Event-based / unspecified'} ({item.get('deadline_type')})")
                    if item.get("dependency_type") not in (None,"none"):
                        st.write(f"Dependency: {item.get('dependency_type')} — {item.get('dependency_text') or ''}")
                    if item.get("condition_text"):
                        st.write(f"Condition: {item['condition_text']}")
                    st.caption(f"Evidence: {item.get('evidence','')}")
                    validation = item['validation']
                    st.write(f"Validation: **{validation['status']}**")
                    for warning in validation.get('warnings', []): st.warning(warning)
                    for error in validation.get('errors', []): st.error(error)
                    opts = ["reject", "skip"] if validation['status'] == 'blocked' else ["reject", "skip", "approve"]
                    choices[index] = st.radio("Decision", options=opts, index=0, horizontal=True, key=f"decision_{index}")
            go = st.form_submit_button("Submit decisions", type="primary")
        if go:
            resume(choices)

    elif request:
        st.error("Unsupported pause request. Do not resume this workflow.")
    else:
        status = state.get("status", "unknown")
        st.subheader("Workflow result")
        st.write("Status:", status)
        if status == "saved":
            saved = state.get("save_result", {})
            st.success(f"Saved {saved.get('saved', 0)} task(s); skipped {saved.get('skipped', 0)} duplicates.")
        elif status == "security_review_required":
            st.error("Transcript requires security review; nothing was saved.")
        elif status == "no_tasks_found":
            st.info("No confirmed tasks found.")
        elif status == "review_completed":
            st.info("No tasks approved, so nothing was saved.")
        else:
            st.info("Workflow completed or interrupted. Review status and logs.")

st.divider()
with st.expander("Saved tasks"):
    try:
        rows = get_tasks()
        if rows:
            visible = [{k: row.get(k) for k in (
                "id", "title", "owner", "deadline_date", "deadline_type",
                "dependency_type", "condition_text", "status"
            )} for row in rows]
            st.dataframe(visible, use_container_width=True, hide_index=True)
        else:
            st.caption("No saved tasks yet")
    except Exception:
        st.error("Unable to load saved tasks")
