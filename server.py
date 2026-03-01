"""
Flask API Server for the Research Agent.
Wraps the LangGraph research pipeline with REST endpoints
for the premium web UI.
"""

import os
import sys
import json
import threading
import io
import time
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Ensure the research_agent module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_agent import run_research, create_empty_knowledge_graph

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ─── Global Mission State ────────────────────────────────────────────────────
mission_state = {
    "status": "idle",           # idle | running | complete | error
    "query": None,
    "search_depth": "advanced",
    "progress": 0,              # 0-100
    "logs": [],
    "knowledge_graph": None,
    "report_markdown": None,
    "whitepaper_markdown": None,
    "sources": [],
    "contradictions": [],
    "started_at": None,
    "completed_at": None,
    "error": None,
}
state_lock = threading.Lock()


def _reset_state():
    """Flush all mission state — the Hard-Reset."""
    global mission_state
    with state_lock:
        mission_state = {
            "status": "idle",
            "query": None,
            "search_depth": "advanced",
            "progress": 0,
            "logs": [],
            "knowledge_graph": None,
            "report_markdown": None,
            "whitepaper_markdown": None,
            "sources": [],
            "contradictions": [],
            "started_at": None,
            "completed_at": None,
            "error": None,
        }


def _append_log(msg: str):
    """Thread-safe log append."""
    with state_lock:
        mission_state["logs"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "message": msg.strip()
        })


def _run_research_thread(query: str, search_depth: str):
    """Run the research pipeline in a background thread, capturing print output."""
    global mission_state

    # Intercept print statements
    old_stdout = sys.stdout
    captured = io.StringIO()

    class TeeOutput:
        """Write to both the captured buffer and real stdout."""
        def __init__(self, captured_io, real_stdout):
            self.captured = captured_io
            self.real = real_stdout
            self._buffer = ""

        def write(self, text):
            self.real.write(text)
            self._buffer += text
            # Flush complete lines to logs
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    _append_log(line)

        def flush(self):
            self.real.flush()
            if self._buffer.strip():
                _append_log(self._buffer)
                self._buffer = ""

    sys.stdout = TeeOutput(captured, old_stdout)

    try:
        with state_lock:
            mission_state["status"] = "running"
            mission_state["progress"] = 5
            mission_state["started_at"] = datetime.now().isoformat()

        _append_log("🚀 Mission started: " + query)

        # Run the research pipeline
        result = run_research(query, search_depth)

        # Extract results
        kg = result.get("knowledge_graph")
        report_md = None
        whitepaper_md = None
        sources = []
        contradictions = []

        # Read the generated report
        report_path = os.path.join("artifacts", "research_report.md")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_md = f.read()

        # Read the whitepaper if it exists
        for fname in os.listdir("artifacts"):
            if fname.startswith("whitepaper_") and fname.endswith(".md"):
                wp_path = os.path.join("artifacts", fname)
                with open(wp_path, "r", encoding="utf-8") as f:
                    whitepaper_md = f.read()
                break

        # Extract sources from KG
        if kg:
            all_urls = set()
            for entity in kg.get("entities", {}).values():
                for url in entity.get("source_urls", []):
                    if url.startswith("http"):
                        all_urls.add(url)
            sources = [{"id": i + 1, "url": url} for i, url in enumerate(sorted(all_urls))]
            contradictions = kg.get("contradictions", [])

        with state_lock:
            mission_state["status"] = "complete"
            mission_state["progress"] = 100
            mission_state["knowledge_graph"] = kg
            mission_state["report_markdown"] = report_md
            mission_state["whitepaper_markdown"] = whitepaper_md
            mission_state["sources"] = sources
            mission_state["contradictions"] = contradictions
            mission_state["completed_at"] = datetime.now().isoformat()

        _append_log("✨ Mission complete!")

    except Exception as e:
        with state_lock:
            mission_state["status"] = "error"
            mission_state["error"] = str(e)
        _append_log(f"❌ Error: {str(e)}")

    finally:
        sys.stdout = old_stdout


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")


@app.route("/api/research", methods=["POST"])
def start_research():
    """Start a new research mission."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    search_depth = data.get("search_depth", "advanced")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    if mission_state["status"] == "running":
        return jsonify({"error": "A mission is already running"}), 409

    # Hard-reset before new mission
    _reset_state()

    with state_lock:
        mission_state["query"] = query
        mission_state["search_depth"] = search_depth

    # Launch in background thread
    thread = threading.Thread(
        target=_run_research_thread,
        args=(query, search_depth),
        daemon=True
    )
    thread.start()

    return jsonify({"status": "started", "query": query})


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get current mission status, logs, and progress."""
    since = request.args.get("since", 0, type=int)

    with state_lock:
        response = {
            "status": mission_state["status"],
            "query": mission_state["query"],
            "progress": mission_state["progress"],
            "logs": mission_state["logs"][since:],
            "log_offset": len(mission_state["logs"]),
            "started_at": mission_state["started_at"],
            "completed_at": mission_state["completed_at"],
            "error": mission_state["error"],
        }

    return jsonify(response)


@app.route("/api/results", methods=["GET"])
def get_results():
    """Get the full research results (KG, report, sources)."""
    with state_lock:
        if mission_state["status"] != "complete":
            return jsonify({"error": "No results available yet"}), 404

        # Build clean entity list (filter word-salad)
        kg = mission_state["knowledge_graph"]
        entities = []
        relationships = []
        word_salad = {"yes", "no", "because", "in", "the", "a", "an", "it",
                      "is", "are", "was", "were", "be", "been", "being",
                      "byte", "default", "full", "low", "on", "off", "max"}

        if kg:
            for eid, e in kg.get("entities", {}).items():
                name = e.get("name", "")
                if name.lower() in word_salad or len(name) < 2:
                    continue
                entities.append({
                    "id": eid,
                    "name": name,
                    "type": e.get("type", "Unknown"),
                    "description": e.get("description", "")[:200],
                    "confidence": e.get("confidence", 0),
                    "sources": e.get("source_urls", []),
                })

            for rel in kg.get("relationships", []):
                src = kg["entities"].get(rel.get("source_id"), {})
                tgt = kg["entities"].get(rel.get("target_id"), {})
                if src.get("name", "").lower() in word_salad:
                    continue
                if tgt.get("name", "").lower() in word_salad:
                    continue
                relationships.append({
                    "source": src.get("name", "?"),
                    "target": tgt.get("name", "?"),
                    "type": rel.get("type", "relates_to"),
                })

        # Build contradictions list
        contradictions = []
        for c in mission_state.get("contradictions", []):
            entity_name = "Unknown"
            if kg:
                entity_name = kg["entities"].get(
                    c.get("entity_id"), {}
                ).get("name", "Unknown")
            contradictions.append({
                "entity": entity_name,
                "claim_a": c.get("claim_a", ""),
                "claim_b": c.get("claim_b", ""),
                "source_a": c.get("source_a", ""),
                "source_b": c.get("source_b", ""),
                "severity": c.get("severity", 0),
                "resolved": c.get("resolved", False),
                "resolution": c.get("resolution", ""),
            })

        return jsonify({
            "entities": entities,
            "relationships": relationships,
            "contradictions": contradictions,
            "report_markdown": mission_state["report_markdown"],
            "whitepaper_markdown": mission_state["whitepaper_markdown"],
            "sources": mission_state["sources"],
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "contradiction_count": len(contradictions),
        })


@app.route("/api/reset", methods=["POST"])
def hard_reset():
    """Hard-Reset: flush all context between missions."""
    if mission_state["status"] == "running":
        return jsonify({"error": "Cannot reset while a mission is running"}), 409

    _reset_state()
    _append_log("🔄 Context flushed — ready for new mission")

    return jsonify({"status": "reset", "message": "All state cleared"})


@app.route("/api/citation/<int:source_id>", methods=["GET"])
def get_citation_detail(source_id: int):
    """Get detailed info for a citation, including which KG nodes reference it."""
    with state_lock:
        kg = mission_state["knowledge_graph"]
        sources = mission_state["sources"]

    if not kg or not sources:
        return jsonify({"error": "No data available"}), 404

    # Find the source URL
    source_url = None
    for s in sources:
        if s["id"] == source_id:
            source_url = s["url"]
            break

    if not source_url:
        return jsonify({"error": f"Source {source_id} not found"}), 404

    # Find all entities that cite this source
    citing_entities = []
    for eid, e in kg.get("entities", {}).items():
        for url in e.get("source_urls", []):
            if source_url in url or url in source_url:
                citing_entities.append({
                    "id": eid,
                    "name": e.get("name", ""),
                    "type": e.get("type", ""),
                })
                break

    return jsonify({
        "source_id": source_id,
        "url": source_url,
        "citing_entities": citing_entities,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 ACE Research Agent — Web UI")
    print("   http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
