"""
Vireo Audio -- Support Analytics dashboard (single-file Flask app).

Reads the real pipeline outputs from ./outputs on every request (mtime-based
cache, so edited files show up on the next refresh), never uses a database,
paid API or LLM, and never hard-codes analysis numbers.

Local:   python app.py            -> http://127.0.0.1:10000
Render:  gunicorn app:app         (respects the PORT env var when run directly)

Routes:  GET /  GET /api/dashboard  GET /api/tickets
         POST /api/refresh  GET /api/refresh/status
"""
import json
import math
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, request

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
BILLING_CATEGORY = "Billing & Payments"   # taxonomy label used by category_tag_audit
PIPELINE_TIMEOUT_SEC = 1800

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# --------------------------------------------------------------------------
# Errors + JSON helpers
# --------------------------------------------------------------------------
class DataError(Exception):
    """A missing / malformed output file. Message is safe to show the user."""


def sanitize(obj):
    """Recursively convert pandas/NumPy values to JSON-safe Python types.
    NaN / NaT / +-inf become None; timestamps become strings."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return None if pd.isna(obj) else str(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (int, str)):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def jresp(payload, status=200):
    body = json.dumps(sanitize(payload), allow_nan=False)
    resp = Response(body, status=status, mimetype="application/json")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


def jerror(message, status=500):
    return jresp({"error": str(message), "status": status}, status)


@app.errorhandler(DataError)
def _handle_data_error(exc):
    return jerror(str(exc), 500)


@app.errorhandler(404)
def _handle_404(exc):
    if request.path.startswith("/api/"):
        return jerror("Not found.", 404)
    return Response("Not found", status=404, mimetype="text/plain")


@app.errorhandler(405)
def _handle_405(exc):
    if request.path.startswith("/api/"):
        return jerror("Method not allowed.", 405)
    return Response("Method not allowed", status=405, mimetype="text/plain")


@app.errorhandler(Exception)
def _handle_any(exc):
    print(f"[app] unhandled error on {request.path}: {exc!r}", file=sys.stderr)
    if request.path.startswith("/api/"):
        return jerror("Unexpected server error while building the response.", 500)
    return Response("Internal server error", status=500, mimetype="text/plain")


# --------------------------------------------------------------------------
# File loading (fresh on change; never cached forever)
# --------------------------------------------------------------------------
_cache = {}
_cache_lock = threading.Lock()

CSV_REQUIRED = {
    "monthly_category.csv": ["month", "predicted_category", "ticket_count"],
    "monthly_team.csv": ["month", "assigned_team", "ticket_count"],
    "team_volume_summary.csv": ["assigned_team", "ticket_count", "share_of_total"],
    "team_workload_summary.csv": [
        "assigned_team", "n_tickets", "median_handle_time_hours",
        "mean_handle_time_hours", "transfer_rate", "high_priority_share",
        "active_agents",
    ],
    "tickets_per_agent.csv": ["assigned_team", "ticket_count", "active_agents", "tickets_per_agent"],
    "categorized_tickets.csv": [
        "ticket_id", "created_at", "channel", "customer_id", "order_id", "category",
        "predicted_category", "category_changed", "priority", "assigned_team",
        "agent_id", "transfers", "csat_score", "source_system", "invalid_date_order",
    ],
}
JSON_FILES = ["business_impact.json", "validation_report.json"]
ALL_OUTPUT_FILES = list(CSV_REQUIRED) + JSON_FILES


def _stat_key(path):
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


def _cached(name, loader):
    path = OUTPUT_DIR / name
    if not path.is_file():
        raise DataError(f"Required output file '{name}' is missing. Run the analysis (Refresh Analysis) first.")
    key = _stat_key(path)
    with _cache_lock:
        hit = _cache.get(name)
        if hit and hit[0] == key:
            return hit[1]
    value = loader(path)
    with _cache_lock:
        _cache[name] = (key, value)
    return value


def load_csv(name):
    required = CSV_REQUIRED[name]

    def loader(path):
        try:
            if name == "categorized_tickets.csv":
                df = pd.read_csv(path, usecols=lambda c: c in set(required) | {"created_at"},
                                 dtype=str, keep_default_na=False, na_values=[""])
            else:
                df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=required)
        except Exception:
            raise DataError(f"Output file '{name}' could not be parsed as CSV.")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataError(f"Output file '{name}' is missing required column(s): {', '.join(missing)}.")
        return df

    return _cached(name, loader).copy()


def load_json(name):
    def loader(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            raise DataError(f"Output file '{name}' is not valid JSON.")
        if not isinstance(data, dict):
            raise DataError(f"Output file '{name}' must contain a JSON object.")
        return data

    return _cached(name, loader)


def load_tickets():
    """categorized_tickets.csv with typed columns. Only whitelisted columns are
    ever read, so raw customer_message / agent_notes text cannot leak."""
    def build():
        df = load_csv("categorized_tickets.csv")
        for c in ["ticket_id", "customer_id", "order_id", "agent_id", "created_at",
                  "channel", "category", "predicted_category", "priority",
                  "assigned_team", "source_system"]:
            df[c] = df[c].astype("object")
        df["category_changed"] = df["category_changed"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        df["invalid_date_order"] = df["invalid_date_order"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        df["transfers"] = pd.to_numeric(df["transfers"], errors="coerce")
        df["csat_score"] = pd.to_numeric(df["csat_score"], errors="coerce")
        return df

    # cache the typed frame against the file's stat key
    path = OUTPUT_DIR / "categorized_tickets.csv"
    if not path.is_file():
        raise DataError("Required output file 'categorized_tickets.csv' is missing. Run the analysis (Refresh Analysis) first.")
    key = _stat_key(path)
    with _cache_lock:
        hit = _cache.get("__typed_tickets__")
        if hit and hit[0] == key:
            return hit[1]
    df = build()
    with _cache_lock:
        _cache["__typed_tickets__"] = (key, df)
    return df


def last_refresh_iso():
    latest = None
    for name in ALL_OUTPUT_FILES:
        p = OUTPUT_DIR / name
        if p.is_file():
            m = p.stat().st_mtime
            latest = m if latest is None or m > latest else latest
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Payload builders
# --------------------------------------------------------------------------
def _records(df):
    return df.to_dict(orient="records")


def _f(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def normalize_business_impact(bi):
    by_channel = bi.get("by_channel", [])
    channels = []
    if isinstance(by_channel, dict):
        for name, row in by_channel.items():
            if isinstance(row, dict):
                channels.append({"channel": name, **row})
    elif isinstance(by_channel, list):
        channels = [r for r in by_channel if isinstance(r, dict)]
    channels = [{
        "channel": str(r.get("channel", "")),
        "n_tickets": _f(r.get("n_tickets")),
        "n_breached": _f(r.get("n_breached")),
        "breach_rate": _f(r.get("breach_rate")),
        "target_minutes": r.get("target_minutes"),
    } for r in channels]

    scenarios = []
    raw = bi.get("scenarios", {})
    if isinstance(raw, dict):
        for key, val in raw.items():
            if isinstance(val, dict) and any(k in val for k in (
                    "annual_savings_inr", "annual_breaches_avoided", "breaches_avoided_per_year")):
                scenarios.append({
                    "key": key,
                    "label": val.get("label") or "Illustrative scenario",
                    "target_rate": val.get("target_rate"),
                    "breaches_avoided": val.get("annual_breaches_avoided", val.get("breaches_avoided_per_year")),
                    "annual_savings_inr": val.get("annual_savings_inr"),
                    "quarterly_savings_inr": val.get("quarterly_savings_inr"),
                })
    audit = bi.get("category_tag_audit", {})
    audit = audit if isinstance(audit, dict) else {}
    table = audit.get("category_delta_table", [])
    table = [r for r in table if isinstance(r, dict)] if isinstance(table, list) else []
    tagged = _f(audit.get("billing_tagged_total"))
    confirmed = _f(audit.get("billing_confirmed_billing"))
    reclassified = audit.get("billing_reclassified_to_other_category")
    reclassified = _f(reclassified) if reclassified is not None else max(tagged - confirmed, 0.0)
    to_delivery = _f(audit.get("billing_reclassified_to_delivery_specifically"))
    return {
        "policy_reference": bi.get("policy_reference"),
        "measured_vs_estimated": bi.get("measured_vs_estimated"),
        "data_window": bi.get("data_window"),
        "overall_breach_rate": _f(bi.get("overall_breach_rate")),
        "n_tickets_in_window": _f(bi.get("n_tickets_in_window")),
        "n_breached_in_window": _f(bi.get("n_breached_in_window")),
        "by_channel": channels,
        "annualized_tickets": _f(bi.get("annualized_tickets")),
        "annualized_breaches": _f(bi.get("annualized_breaches")),
        "sla_breach_credit_inr": _f(bi.get("sla_breach_credit_inr")),
        "current_annual_liability_inr": _f(bi.get("current_annual_liability_inr")),
        "current_quarterly_liability_inr": _f(bi.get("current_quarterly_liability_inr")),
        "scenarios": scenarios,
        "scenario_assumption": (raw.get("assumption") if isinstance(raw, dict) else None),
        "category_tag_audit": {
            "overall_tag_change_rate": audit.get("overall_tag_change_rate"),
            "category_delta_table": [{
                "category": str(r.get("category", "")),
                "bot_tagged_count": _f(r.get("bot_tagged_count")),
                "reclassified_count": _f(r.get("reclassified_count")),
                "delta": _f(r.get("delta")),
                "pct_change": r.get("pct_change"),
            } for r in table],
            "billing_tagged_total": tagged,
            "billing_confirmed_billing": confirmed,
            "billing_reclassified": reclassified,
            "billing_reclassified_to_delivery": to_delivery,
            "billing_reclassified_to_other_non_delivery": max(reclassified - to_delivery, 0.0),
            "billing_reclassified_pct": (reclassified / tagged * 100.0) if tagged else None,
        },
    }


def normalize_validation(v):
    per = v.get("per_category_sample_precision", {})
    per_rows = []
    if isinstance(per, dict):
        for cat, row in per.items():
            row = row if isinstance(row, dict) else {}
            per_rows.append({
                "category": str(cat),
                "n_predicted": row.get("n_predicted_in_sample"),
                "n_errors": row.get("n_errors"),
                "precision": row.get("sample_precision"),
            })
    elif isinstance(per, list):
        for row in per:
            if isinstance(row, dict):
                per_rows.append({
                    "category": str(row.get("category", "")),
                    "n_predicted": row.get("n_predicted_in_sample", row.get("n_predicted")),
                    "n_errors": row.get("n_errors"),
                    "precision": row.get("sample_precision", row.get("precision")),
                })
    fm = v.get("failure_modes", {})
    fm_rows = []
    if isinstance(fm, dict):
        fm_rows = [{"mode": str(k), "count": val} for k, val in fm.items()]
    elif isinstance(fm, list):
        fm_rows = [{"mode": str(r.get("mode", r)) if isinstance(r, dict) else str(r),
                    "count": r.get("count") if isinstance(r, dict) else None} for r in fm]
    errs = v.get("errors_detail", [])
    errs = [e for e in errs if isinstance(e, dict)] if isinstance(errs, list) else []
    return {
        "method": v.get("method"),
        "sample_size": v.get("sample_size"),
        "random_seed": v.get("random_seed"),
        "taxonomy_version": v.get("taxonomy_version"),
        "model": v.get("model"),
        "reviewer_type": v.get("reviewer_type"),
        "accuracy": v.get("accuracy"),
        "n_correct": v.get("n_correct"),
        "n_errors": v.get("n_errors"),
        "error_rate": v.get("error_rate"),
        "per_category_precision": per_rows,
        "failure_modes": fm_rows,
        "errors_detail": errs,
        "limitation": v.get("limitation"),
    }


def build_team_summary(volume, workload, per_agent):
    teams = {}
    for _, r in volume.iterrows():
        teams.setdefault(r["assigned_team"], {})["ticket_count"] = r["ticket_count"]
        teams[r["assigned_team"]]["share_of_total"] = r["share_of_total"]
    for _, r in per_agent.iterrows():
        t = teams.setdefault(r["assigned_team"], {})
        t["active_agents"] = r["active_agents"]
        t["tickets_per_agent"] = r["tickets_per_agent"]
        t.setdefault("ticket_count", r["ticket_count"])
    for _, r in workload.iterrows():
        t = teams.setdefault(r["assigned_team"], {})
        t["n_tickets_workload"] = r["n_tickets"]
        t["median_handle_time_hours"] = r["median_handle_time_hours"]
        t["mean_handle_time_hours"] = r["mean_handle_time_hours"]
        t["transfer_rate"] = r["transfer_rate"]
        t["high_priority_share"] = r["high_priority_share"]
        t.setdefault("active_agents", r["active_agents"])
    rows = [{"assigned_team": k, **v} for k, v in teams.items()]
    rows.sort(key=lambda r: -_f(r.get("ticket_count")))
    return rows


def build_category_summary(tickets):
    total = len(tickets)
    orig = tickets["category"].fillna("(blank)").value_counts()
    pred = tickets["predicted_category"].fillna("(blank)").value_counts()
    changed_from = tickets[tickets["category_changed"]]["category"].fillna("(blank)").value_counts()
    rows = []
    for cat in sorted(set(orig.index) | set(pred.index)):
        o, p = int(orig.get(cat, 0)), int(pred.get(cat, 0))
        cf = int(changed_from.get(cat, 0))
        rows.append({
            "category": cat,
            "original_count": o,
            "predicted_count": p,
            "delta": p - o,
            "share_of_total_predicted": (p / total) if total else None,
            "original_reclassified_away": cf,
            "original_reclassified_away_rate": (cf / o) if o else None,
        })
    rows.sort(key=lambda r: -r["predicted_count"])
    return rows


def build_transitions(tickets, limit=15):
    ch = tickets[tickets["category_changed"]]
    if ch.empty:
        return []
    g = (ch.groupby(["category", "predicted_category"]).size()
           .reset_index(name="ticket_count").sort_values("ticket_count", ascending=False).head(limit))
    return [{"original_category": r["category"], "predicted_category": r["predicted_category"],
             "ticket_count": int(r["ticket_count"])} for _, r in g.iterrows()]


def build_billing_breakdown(tickets):
    b = tickets[tickets["category"] == BILLING_CATEGORY]
    if b.empty:
        return []
    vc = b["predicted_category"].fillna("(blank)").value_counts()
    total = int(vc.sum())
    return [{"predicted_category": k, "ticket_count": int(v), "share": v / total} for k, v in vc.items()]


def build_filter_options(tickets):
    def opts(col):
        return sorted(str(x) for x in tickets[col].dropna().unique())
    return {
        "original_category": opts("category"),
        "predicted_category": opts("predicted_category"),
        "team": opts("assigned_team"),
        "channel": opts("channel"),
        "priority": opts("priority"),
        "source_system": opts("source_system"),
    }


def build_dashboard():
    tickets = load_tickets()
    monthly_category = load_csv("monthly_category.csv")
    monthly_team = load_csv("monthly_team.csv")
    team_volume = load_csv("team_volume_summary.csv")
    team_workload = load_csv("team_workload_summary.csv")
    per_agent = load_csv("tickets_per_agent.csv")
    bi = normalize_business_impact(load_json("business_impact.json"))
    validation = normalize_validation(load_json("validation_report.json"))

    total = int(len(tickets))
    changed = int(tickets["category_changed"].sum()) if total else 0
    created = tickets["created_at"].dropna().astype(str)
    active_agents = int(pd.to_numeric(per_agent["active_agents"], errors="coerce").fillna(0).sum())
    if active_agents == 0:
        active_agents = int(pd.to_numeric(team_workload["active_agents"], errors="coerce").fillna(0).sum())

    overview = {
        "total_tickets": total,
        "categories_changed": changed,
        "category_change_rate": (changed / total) if total else None,
        "active_agents": active_agents,
        "invalid_date_order": int(tickets["invalid_date_order"].sum()) if total else 0,
        "date_min": created.min() if len(created) else None,
        "date_max": created.max() if len(created) else None,
        "source_system_counts": {str(k): int(v) for k, v in tickets["source_system"].fillna("(blank)").value_counts().items()},
    }
    return {
        "overview": overview,
        "monthly_category": _records(monthly_category),
        "monthly_team": _records(monthly_team),
        "team_volume": _records(team_volume),
        "team_workload": _records(team_workload),
        "tickets_per_agent": _records(per_agent),
        "team_summary": build_team_summary(team_volume, team_workload, per_agent),
        "business_impact": bi,
        "validation": validation,
        "category_summary": build_category_summary(tickets),
        "category_transitions": build_transitions(tickets),
        "billing_breakdown": build_billing_breakdown(tickets),
        "filter_options": build_filter_options(tickets),
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "last_refresh": last_refresh_iso(),
            "files": ALL_OUTPUT_FILES,
        },
    }


# --------------------------------------------------------------------------
# Refresh (runs `python run.py` in the background)
# --------------------------------------------------------------------------
_refresh = {"running": False, "state": "idle", "message": "No refresh has been run in this session.",
            "started_at": None, "finished_at": None}
_refresh_lock = threading.Lock()


def _run_pipeline():
    try:
        script = BASE_DIR / "run.py"
        if not script.is_file():
            raise FileNotFoundError("run.py not found")
        proc = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR),
                              capture_output=True, text=True, timeout=PIPELINE_TIMEOUT_SEC)
        if proc.returncode == 0:
            state, msg = "success", "Analysis pipeline finished successfully."
        else:
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            print(f"[refresh] pipeline failed (exit {proc.returncode}):\n{tail}", file=sys.stderr)
            state, msg = "error", f"Pipeline failed (exit code {proc.returncode}). See server logs."
    except subprocess.TimeoutExpired:
        state, msg = "error", "Pipeline timed out and was stopped."
    except FileNotFoundError:
        state, msg = "error", "run.py was not found next to app.py."
    except Exception as exc:  # noqa: BLE001
        print(f"[refresh] unexpected error: {exc!r}", file=sys.stderr)
        state, msg = "error", "Unexpected error while running the pipeline. See server logs."
    with _refresh_lock:
        _refresh.update(running=False, state=state, message=msg,
                        finished_at=datetime.now(timezone.utc).isoformat())


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    with _refresh_lock:
        if _refresh["running"]:
            return jresp({"status": "already_running"})
        _refresh.update(running=True, state="running", message="Analysis pipeline is running...",
                        started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return jresp({"status": "started"})


@app.route("/api/refresh/status")
def api_refresh_status():
    with _refresh_lock:
        snap = dict(_refresh)
    return jresp({"running": snap["running"], "state": snap["state"], "message": snap["message"],
                  "started_at": snap["started_at"], "finished_at": snap["finished_at"],
                  "last_refresh": last_refresh_iso()})


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------
@app.route("/api/dashboard")
def api_dashboard():
    return jresp(build_dashboard())


TICKET_COLUMNS = ["ticket_id", "created_at", "channel", "category", "predicted_category",
                  "category_changed", "priority", "assigned_team", "agent_id", "transfers",
                  "csat_score", "source_system", "invalid_date_order", "customer_id", "order_id"]
SORTABLE = set(TICKET_COLUMNS)


def _arg(name, maxlen=100):
    v = request.args.get(name, "", type=str) or ""
    return v.strip()[:maxlen]


def _int_arg(name, default, lo, hi):
    try:
        v = int(request.args.get(name, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


@app.route("/api/tickets")
def api_tickets():
    df = load_tickets()
    page = _int_arg("page", 1, 1, 10**7)
    page_size = _int_arg("page_size", 50, 1, 200)

    search = _arg("search").lower()
    if search:
        mask = pd.Series(False, index=df.index)
        for col in ("ticket_id", "customer_id", "order_id"):
            mask |= df[col].fillna("").astype(str).str.lower().str.contains(search, regex=False)
        df = df[mask]

    for param, col in (("original_category", "category"), ("predicted_category", "predicted_category"),
                       ("team", "assigned_team"), ("channel", "channel"),
                       ("priority", "priority"), ("source_system", "source_system")):
        val = _arg(param)
        if val and val.lower() != "all":
            df = df[df[col] == val]

    changed = _arg("changed").lower()
    if changed in ("true", "yes", "1"):
        df = df[df["category_changed"]]
    elif changed in ("false", "no", "0"):
        df = df[~df["category_changed"]]

    sort_by = _arg("sort_by") or "created_at"
    if sort_by not in SORTABLE:
        sort_by = "created_at"
    ascending = (_arg("sort_dir").lower() == "asc")
    df = df.sort_values(sort_by, ascending=ascending, kind="mergesort", na_position="last")

    total = int(len(df))
    pages = max(1, math.ceil(total / page_size))
    page = min(page, pages)
    chunk = df.iloc[(page - 1) * page_size: page * page_size][TICKET_COLUMNS]
    return jresp({"rows": _records(chunk), "page": page, "page_size": page_size,
                  "total": total, "pages": pages, "sort_by": sort_by,
                  "sort_dir": "asc" if ascending else "desc"})


@app.route("/")
def index():
    resp = Response(INDEX_HTML, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


# --------------------------------------------------------------------------
# Front end (plain HTML/CSS/JS + Chart.js CDN, no build step)
# --------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vireo Audio · Support Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--navy:#0b1b34;--navy2:#13294b;--accent:#2563eb;--accent-soft:#e8f0fe;--bg:#f3f6fb;--card:#fff;--line:#e2e8f0;
--text:#0f172a;--muted:#64748b;--good:#15803d;--warn:#b45309;--bad:#b91c1c;--shadow:0 1px 2px rgba(15,23,42,.06),0 4px 14px rgba(15,23,42,.05)}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.sidebar{position:fixed;inset:0 auto 0 0;width:240px;background:linear-gradient(180deg,var(--navy),var(--navy2));color:#cbd5e1;display:flex;flex-direction:column;z-index:30;transition:transform .2s}
.brand{padding:22px 20px 16px;border-bottom:1px solid rgba(255,255,255,.08)}
.brand b{display:block;color:#fff;font-size:18px;letter-spacing:.3px}
.brand span{font-size:12px;color:#8ea2c4}
.nav{padding:12px 10px;display:flex;flex-direction:column;gap:2px;overflow:auto}
.nav button{all:unset;cursor:pointer;padding:10px 12px;border-radius:8px;color:#cbd5e1;display:flex;gap:10px;align-items:center;font-size:14px}
.nav button:hover{background:rgba(255,255,255,.07);color:#fff}
.nav button.active{background:var(--accent);color:#fff;font-weight:600}
.nav .ic{width:18px;text-align:center;opacity:.9}
.side-foot{margin-top:auto;padding:14px 18px;font-size:11px;color:#7f93b5;border-top:1px solid rgba(255,255,255,.08)}
.main{margin-left:240px;min-height:100vh}
.header{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.92);backdrop-filter:blur(6px);border-bottom:1px solid var(--line);
display:flex;align-items:center;gap:14px;padding:12px 24px;flex-wrap:wrap}
.header h1{font-size:17px;margin:0}
.header .sub{color:var(--muted);font-size:12px}
.spacer{flex:1}
.stamp{font-size:12px;color:var(--muted);text-align:right}
.stamp b{color:var(--text)}
.btn{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 16px;font-weight:600;cursor:pointer;font-size:13px}
.btn:hover{background:#1d4ed8}.btn:disabled{opacity:.6;cursor:not-allowed}
.btn.ghost{background:#fff;color:var(--text);border:1px solid var(--line)}
.spinner{width:16px;height:16px;border:2px solid #cbd5e1;border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;display:none}
.spinner.on{display:inline-block}@keyframes spin{to{transform:rotate(360deg)}}
.burger{display:none;background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer;font-size:16px}
.content{padding:22px 24px 60px;max-width:1500px}
.page{display:none}.page.active{display:block}
h2.title{margin:2px 0 4px;font-size:20px}
p.lead{margin:0 0 18px;color:var(--muted)}
.grid{display:grid;gap:16px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(200px,1fr));margin-bottom:18px}
.cols2{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:16px 18px;min-width:0}
.card h3{margin:0 0 10px;font-size:14px}
.card h3 small{font-weight:400;color:var(--muted);margin-left:6px}
.kpi .label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
.kpi .value{font-size:26px;font-weight:700;margin:4px 0 2px}
.kpi .subtxt{font-size:12px;color:var(--muted)}
.kpi.warn{border-left:4px solid var(--warn)}.kpi.good{border-left:4px solid var(--good)}.kpi.blue{border-left:4px solid var(--accent)}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;vertical-align:middle}
.b-meas{background:#dcfce7;color:#166534}.b-est{background:#fef3c7;color:#92400e}.b-ai{background:#ede9fe;color:#5b21b6}
.b-red{background:#fee2e2;color:#991b1b}.b-gray{background:#e2e8f0;color:#334155}.b-blue{background:var(--accent-soft);color:#1d4ed8}
.chartbox{position:relative;height:320px}.chartbox.tall{height:380px}
.chartbox canvas{max-width:100%}
.chartbox .empty{display:none;position:absolute;inset:0;align-items:center;justify-content:center;color:var(--muted);font-size:13px}
.notice{border-radius:10px;padding:12px 14px;margin:0 0 16px;font-size:13px;border:1px solid}
.n-warn{background:#fffbeb;border-color:#fcd34d;color:#78350f}
.n-ai{background:#f5f3ff;border-color:#c4b5fd;color:#4c1d95}
.n-info{background:#eff6ff;border-color:#bfdbfe;color:#1e3a8a}
.n-err{background:#fef2f2;border-color:#fca5a5;color:#7f1d1d}
.n-ok{background:#f0fdf4;border-color:#86efac;color:#14532d}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:#f8fafc;font-size:12px;color:#475569;text-transform:uppercase;letter-spacing:.3px;position:sticky;top:0}
th.sortable{cursor:pointer;user-select:none}th.sortable:hover{color:var(--accent)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}tbody tr:hover{background:#f8fafc}
td.emptyrow{text-align:center;color:var(--muted);padding:24px;white-space:normal}
ul.findings{margin:0;padding-left:18px}ul.findings li{margin:6px 0}
.filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:12px}
.filters label{font-size:11px;color:var(--muted);display:block;margin-bottom:3px;text-transform:uppercase;letter-spacing:.3px}
.filters input,.filters select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;font-size:13px}
.pager{display:flex;align-items:center;gap:10px;justify-content:space-between;margin-top:12px;flex-wrap:wrap;color:var(--muted);font-size:13px}
.pager .btns{display:flex;gap:6px;align-items:center}
.small{font-size:12px;color:var(--muted)}
#errbanner{margin:0 0 16px}
.sect-err{color:var(--bad);font-size:13px}
@media(max-width:900px){
 .sidebar{transform:translateX(-100%)}.sidebar.open{transform:none}
 .main{margin-left:0}.burger{display:inline-block}
 .content{padding:16px 14px 50px}.cols2{grid-template-columns:1fr}.header{padding:10px 14px}
}
</style>
</head>
<body>
<aside class="sidebar" id="sidebar">
  <div class="brand"><b>Vireo Audio</b><span>Support Analytics</span></div>
  <nav class="nav" id="nav">
    <button data-page="overview" class="active"><span class="ic">▦</span>Overview</button>
    <button data-page="categories"><span class="ic">◧</span>Categories</button>
    <button data-page="teams"><span class="ic">☷</span>Teams &amp; Workload</button>
    <button data-page="sla"><span class="ic">⏱</span>SLA &amp; Financial</button>
    <button data-page="billing"><span class="ic">₹</span>Billing Audit</button>
    <button data-page="validation"><span class="ic">✓</span>Validation</button>
    <button data-page="explorer"><span class="ic">⌕</span>Ticket Explorer</button>
  </nav>
  <div class="side-foot">All figures are read live from the pipeline's <code>outputs/</code> files.</div>
</aside>

<div class="main">
  <header class="header">
    <button class="burger" id="burger" aria-label="Menu">☰</button>
    <div><h1>Vireo Audio <span class="sub">· Support Analytics</span></h1></div>
    <div class="spacer"></div>
    <div class="stamp">Last data refresh<br><b id="lastRefresh">—</b></div>
    <span class="spinner" id="spinner" title="Loading"></span>
    <button class="btn" id="refreshBtn">Refresh Analysis</button>
  </header>

  <main class="content">
    <div id="errbanner"></div>

    <!-- OVERVIEW -->
    <section class="page active" id="page-overview">
      <h2 class="title">Overview</h2>
      <p class="lead">Headline numbers and trends from the latest analysis run.</p>
      <div class="grid kpis" id="ov-kpis"></div>
      <div id="ov-notes"></div>
      <div class="grid cols2">
        <div class="card"><h3>Key findings <small>generated from the loaded data</small></h3><ul class="findings" id="ov-findings"></ul></div>
        <div class="card"><h3>Monthly ticket volume</h3><div class="chartbox"><canvas id="c-ov-volume"></canvas><div class="empty">No monthly data available.</div></div></div>
        <div class="card"><h3>Monthly category trends <small>rule-based predicted category</small></h3><div class="chartbox tall"><canvas id="c-ov-cat"></canvas><div class="empty">No monthly category data available.</div></div></div>
        <div class="card"><h3>Monthly team trends <small>historical assigned team</small></h3><div class="chartbox tall"><canvas id="c-ov-team"></canvas><div class="empty">No monthly team data available.</div></div></div>
      </div>
    </section>

    <!-- CATEGORIES -->
    <section class="page" id="page-categories">
      <h2 class="title">Categories</h2>
      <p class="lead">Original (bot) category versus the current rule-based <b>predicted</b> category. Predictions are inferences, not verified ground truth.</p>
      <div class="card" style="margin-bottom:16px"><h3>Category volume: original vs predicted</h3><div id="t-cat"></div></div>
      <div class="grid cols2">
        <div class="card"><h3>Predicted category distribution</h3><div class="chartbox tall"><canvas id="c-cat-dist"></canvas><div class="empty">No category data.</div></div></div>
        <div class="card"><h3>Category changes <small>original vs predicted count</small></h3><div class="chartbox tall"><canvas id="c-cat-changes"></canvas><div class="empty">No category data.</div></div></div>
        <div class="card" style="grid-column:1/-1"><h3>Monthly category volume</h3><div class="chartbox tall"><canvas id="c-cat-monthly"></canvas><div class="empty">No monthly category data.</div></div></div>
        <div class="card"><h3>Category audit <small>from business_impact.json</small></h3><div id="t-cat-audit"></div></div>
        <div class="card"><h3>Most common reclassifications <small>original → predicted</small></h3><div id="t-cat-trans"></div></div>
      </div>
    </section>

    <!-- TEAMS -->
    <section class="page" id="page-teams">
      <h2 class="title">Teams &amp; Workload</h2>
      <p class="lead">Volume, staffing and handling measures by historical assigned team.</p>
      <div class="notice n-warn"><b>Read with care:</b> raw ticket volume alone does not establish staffing need. Teams differ in headcount, handle time and how much of their volume is mis-tagged (see Billing Audit). Tickets-per-agent divides all tickets in the dataset by the current active roster, so it is a rough ratio, not a time-matched utilisation figure. This dashboard does not recommend where the next hires should go.</div>
      <div class="card" style="margin-bottom:16px"><h3>Team summary</h3><div id="t-team"></div>
        <p class="small" style="margin:8px 0 0">Handle-time, transfer and priority columns use helpdesk-era resolved tickets with valid timestamps only (n shown in the workload column).</p></div>
      <div class="grid cols2">
        <div class="card"><h3>Team volume</h3><div class="chartbox"><canvas id="c-team-vol"></canvas><div class="empty">No team volume data.</div></div></div>
        <div class="card"><h3>Tickets per active agent</h3><div class="chartbox"><canvas id="c-team-per"></canvas><div class="empty">No tickets-per-agent data.</div></div></div>
        <div class="card" style="grid-column:1/-1"><h3>Monthly team volume</h3><div class="chartbox tall"><canvas id="c-team-monthly"></canvas><div class="empty">No monthly team data.</div></div></div>
      </div>
    </section>

    <!-- SLA -->
    <section class="page" id="page-sla">
      <h2 class="title">SLA &amp; Financial</h2>
      <p class="lead">First-response performance and the policy-based credit exposure.</p>
      <div id="sla-window"></div>
      <h3 style="margin:6px 0 10px">Measured data <span class="badge b-meas">MEASURED</span></h3>
      <div class="grid kpis" id="sla-kpis-measured"></div>
      <h3 style="margin:6px 0 10px">Estimated / annualized financial impact <span class="badge b-est">ESTIMATED</span></h3>
      <div class="grid kpis" id="sla-kpis-est"></div>
      <div id="sla-est-note"></div>
      <div class="card" style="margin-bottom:16px"><h3>By channel <span class="badge b-meas">MEASURED</span></h3><div id="t-sla"></div></div>
      <div class="grid cols2">
        <div class="card"><h3>SLA breach rate by channel</h3><div class="chartbox"><canvas id="c-sla-rate"></canvas><div class="empty">No channel data.</div></div></div>
        <div class="card"><h3>Tickets by channel</h3><div class="chartbox"><canvas id="c-sla-tix"></canvas><div class="empty">No channel data.</div></div></div>
        <div class="card"><h3>Breached tickets by channel</h3><div class="chartbox"><canvas id="c-sla-br"></canvas><div class="empty">No channel data.</div></div></div>
        <div class="card"><h3>Halve-breach-rate scenario <span class="badge b-gray">ILLUSTRATIVE</span></h3><div id="sla-scenario"></div></div>
      </div>
    </section>

    <!-- BILLING -->
    <section class="page" id="page-billing">
      <h2 class="title">Billing Audit</h2>
      <p class="lead">What the rule-based re-read of the ticket text assigns to tickets the intake bot tagged as Billing &amp; Payments. These are text-based reclassifications, not independently verified categories.</p>
      <div class="grid kpis" id="bill-kpis"></div>
      <div class="grid cols2">
        <div class="card"><h3>Billing-tagged tickets: re-read outcome</h3><div class="chartbox tall"><canvas id="c-bill-donut"></canvas><div class="empty">No billing audit data.</div></div></div>
        <div class="card"><h3>Category tag changes <small>bot-tagged vs re-read</small></h3><div class="chartbox tall"><canvas id="c-bill-delta"></canvas><div class="empty">No category audit data.</div></div></div>
        <div class="card"><h3>Billing audit table</h3><div id="t-bill"></div></div>
        <div class="card"><h3>Where Billing-tagged tickets went <small>from categorized_tickets.csv</small></h3><div id="t-bill-break"></div></div>
      </div>
    </section>

    <!-- VALIDATION -->
    <section class="page" id="page-validation">
      <h2 class="title">Validation</h2>
      <div class="notice n-ai"><b>AI-assisted first-pass validation — not an independent human audit.</b> The reference labels were assigned by the same AI assistant that built the classifier, and only the originally-flagged tickets have stored labels, so new errors elsewhere in the sample would not be detected. Treat the figure below as agreement with AI-reviewed labels, not audited accuracy.</div>
      <div class="grid kpis" id="val-kpis"></div>
      <div id="val-meta"></div>
      <div class="grid cols2">
        <div class="card"><h3>Per-category precision <small>in the validation sample</small></h3><div class="chartbox tall"><canvas id="c-val-prec"></canvas><div class="empty">No per-category data.</div></div></div>
        <div class="card"><h3>Failure modes</h3><div id="t-val-fm"></div>
          <h3 style="margin-top:18px">Per-category detail</h3><div id="t-val-prec"></div></div>
        <div class="card" style="grid-column:1/-1"><h3>Detailed validation errors</h3><div id="t-val-err"></div></div>
        <div class="card" style="grid-column:1/-1"><h3>Stated limitation</h3><p id="val-limit" style="margin:0"></p></div>
      </div>
    </section>

    <!-- EXPLORER -->
    <section class="page" id="page-explorer">
      <h2 class="title">Ticket Explorer</h2>
      <p class="lead">Browse categorized tickets. Raw customer messages and agent notes are not loaded or displayed.</p>
      <div class="card">
        <div class="filters" id="ex-filters">
          <div><label for="f-search">Search ticket / customer / order ID</label><input id="f-search" type="search" placeholder="e.g. TK-240001, C103739"></div>
          <div><label for="f-original">Original category</label><select id="f-original"></select></div>
          <div><label for="f-predicted">Predicted category</label><select id="f-predicted"></select></div>
          <div><label for="f-team">Team</label><select id="f-team"></select></div>
          <div><label for="f-channel">Channel</label><select id="f-channel"></select></div>
          <div><label for="f-priority">Priority</label><select id="f-priority"></select></div>
          <div><label for="f-changed">Category changed</label><select id="f-changed"><option value="">All</option><option value="true">Changed</option><option value="false">Not changed</option></select></div>
          <div><label for="f-source">Source system</label><select id="f-source"></select></div>
          <div><label for="f-size">Rows per page</label><select id="f-size"><option>25</option><option selected>50</option><option>100</option><option>200</option></select></div>
          <div style="display:flex;align-items:flex-end"><button class="btn ghost" id="f-reset" style="width:100%">Reset filters</button></div>
        </div>
        <div id="ex-table"></div>
        <div class="pager"><span id="ex-info"></span>
          <span class="btns"><button class="btn ghost" id="ex-prev">‹ Prev</button><span id="ex-page"></span><button class="btn ghost" id="ex-next">Next ›</button></span></div>
      </div>
    </section>
  </main>
</div>

<script>
"use strict";
/* ---------- defensive helpers ---------- */
const $ = (id) => document.getElementById(id);
const asArray = (v) => (Array.isArray(v) ? v : []);
const asObj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
const num = (v, d = 0) => { if (v === null || v === undefined || v === "") return d; const n = Number(v); return Number.isFinite(n) ? n : d; };
const isNil = (v) => v === null || v === undefined || v === "" || (typeof v === "number" && !Number.isFinite(v));
const nf = new Intl.NumberFormat("en-IN");
const nf1 = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const fmtInt = (v) => (isNil(v) ? "—" : nf.format(Math.round(num(v))));
const fmtNum1 = (v) => (isNil(v) ? "—" : nf1.format(num(v)));
const fmtPct = (f, dp = 1) => (isNil(f) ? "—" : (num(f) * 100).toFixed(dp) + "%");
const fmtInr = (v) => (isNil(v) ? "—" : inr.format(num(v)));
const fmtDate = (s) => { if (!s) return "—"; const d = new Date(s); return isNaN(d) ? String(s) : d.toLocaleString("en-IN"); };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const PALETTE = ["#2563eb","#f59e0b","#10b981","#ef4444","#8b5cf6","#06b6d4","#ec4899","#84cc16","#f97316","#64748b","#14b8a6","#a855f7"];
const S = { data: null, charts: {}, ex: { page: 1, sort_by: "created_at", sort_dir: "desc" }, exReady: false, polling: null };

/* ---------- api ---------- */
async function api(url, opts) {
  let res;
  try { res = await fetch(url, Object.assign({ cache: "no-store" }, opts || {})); }
  catch (e) { throw new Error("Network error: could not reach the server."); }
  let body = null;
  try { body = await res.json(); } catch (e) { body = null; }
  if (!res.ok || (body && body.error)) throw new Error((body && body.error) || `Request failed (HTTP ${res.status}).`);
  if (body === null) throw new Error("Server returned an unexpected (non-JSON) response.");
  return body;
}
function showBanner(kind, msg) { $("errbanner").innerHTML = msg ? `<div class="notice ${kind}">${esc(msg)}</div>` : ""; }
function setLoading(on) { $("spinner").classList.toggle("on", !!on); }

/* ---------- generic table ---------- */
function renderTable(elId, cols, rows, opts) {
  const el = $(elId); if (!el) return;
  opts = opts || {};
  rows = asArray(rows);
  const head = cols.map((c) => {
    const sortable = opts.sortable && c.sort;
    const arrow = sortable && opts.sortBy === c.sort ? (opts.sortDir === "asc" ? " ▲" : " ▼") : "";
    return `<th class="${c.num ? "num " : ""}${sortable ? "sortable" : ""}" ${sortable ? `data-sort="${esc(c.sort)}"` : ""}>${esc(c.label)}${arrow}</th>`;
  }).join("");
  let body;
  if (!rows.length) body = `<tr><td class="emptyrow" colspan="${cols.length}">${esc(opts.empty || "No data to display.")}</td></tr>`;
  else body = rows.map((r) => "<tr>" + cols.map((c) => {
    const raw = r ? r[c.key] : undefined;
    const cell = c.html ? c.html(raw, r) : esc(c.fmt ? c.fmt(raw, r) : (isNil(raw) ? "—" : raw));
    return `<td class="${c.num ? "num" : ""}">${cell}</td>`;
  }).join("") + "</tr>").join("");
  el.innerHTML = `<div class="tablewrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  if (opts.onSort) el.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => opts.onSort(th.dataset.sort)));
}

/* ---------- charts ---------- */
function drawChart(canvasId, config, hasData) {
  const canvas = $(canvasId); if (!canvas) return;
  if (S.charts[canvasId]) { S.charts[canvasId].destroy(); delete S.charts[canvasId]; }
  const empty = canvas.parentElement.querySelector(".empty");
  if (!hasData || typeof Chart === "undefined") {
    canvas.style.display = "none";
    if (empty) { empty.style.display = "flex"; if (typeof Chart === "undefined") empty.textContent = "Chart library failed to load (check internet access to the Chart.js CDN)."; }
    return;
  }
  canvas.style.display = ""; if (empty) empty.style.display = "none";
  config.options = Object.assign({ responsive: true, maintainAspectRatio: false }, config.options || {});
  S.charts[canvasId] = new Chart(canvas.getContext("2d"), config);
}
function pivot(rows, seriesKey, valueKey) {
  const months = new Set(), series = {};
  asArray(rows).forEach((r) => {
    const m = String(r?.month ?? ""), s = String(r?.[seriesKey] ?? "(blank)");
    if (!m) return; months.add(m);
    (series[s] = series[s] || {})[m] = num(r?.[valueKey]);
  });
  const labels = [...months].sort();
  return { labels, series: Object.keys(series).sort().map((name, i) => ({
    label: name, data: labels.map((m) => series[name][m] ?? 0),
    borderColor: PALETTE[i % PALETTE.length], backgroundColor: PALETTE[i % PALETTE.length] + "33", tension: .25, pointRadius: 2, borderWidth: 2 })) };
}
function lineChart(id, rows, seriesKey) {
  const p = pivot(rows, seriesKey, "ticket_count");
  drawChart(id, { type: "line", data: { labels: p.labels, datasets: p.series },
    options: { interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } },
      tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtInt(c.parsed.y)}` } } },
      scales: { y: { beginAtZero: true, ticks: { callback: (v) => nf.format(v) }, title: { display: true, text: "Tickets" } } } } }, p.labels.length && p.series.length);
}
function barChart(id, labels, datasets, opts) {
  opts = opts || {};
  labels = asArray(labels);
  const has = labels.length && datasets.some((d) => asArray(d.data).length);
  drawChart(id, { type: "bar", data: { labels, datasets },
    options: { indexAxis: opts.horizontal ? "y" : "x", plugins: { legend: { display: datasets.length > 1, position: "bottom" },
      tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${(opts.fmt || fmtInt)(c.parsed[opts.horizontal ? "x" : "y"])}` } } },
      scales: { [opts.horizontal ? "x" : "y"]: { beginAtZero: true, max: opts.max, ticks: { callback: (v) => (opts.fmt || ((x) => nf.format(x)))(v) } } } } }, has);
}
function doughnut(id, labels, values, colors) {
  const has = asArray(values).some((v) => num(v) > 0);
  drawChart(id, { type: "doughnut", data: { labels, datasets: [{ data: values, backgroundColor: colors || PALETTE, borderWidth: 1 }] },
    options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } },
      tooltip: { callbacks: { label: (c) => { const t = c.dataset.data.reduce((a, b) => a + num(b), 0); return `${c.label}: ${fmtInt(c.parsed)} (${t ? (c.parsed / t * 100).toFixed(1) : 0}%)`; } } } } } }, has);
}

/* ---------- KPI helper ---------- */
const kpi = (label, value, sub, tone, badge) => `<div class="card kpi ${tone || ""}"><div class="label">${esc(label)} ${badge || ""}</div><div class="value">${esc(value)}</div><div class="subtxt">${esc(sub || "")}</div></div>`;
const B = { meas: '<span class="badge b-meas">MEASURED</span>', est: '<span class="badge b-est">ESTIMATED</span>', ai: '<span class="badge b-ai">AI FIRST-PASS</span>' };

/* ---------- section renderers ---------- */
function buildFindings(d) {
  const out = [], ov = asObj(d.overview), bi = asObj(d.business_impact), audit = asObj(bi.category_tag_audit), val = asObj(d.validation);
  const teams = asArray(d.team_summary);
  if (ov.total_tickets) out.push(`${fmtInt(ov.total_tickets)} tickets analysed (${esc0(ov.date_min)} to ${esc0(ov.date_max)}).`);
  if (!isNil(ov.category_change_rate)) out.push(`${fmtInt(ov.categories_changed)} tickets (${fmtPct(ov.category_change_rate)}) have a rule-based predicted category that differs from the original bot category.`);
  if (num(audit.billing_tagged_total) > 0) out.push(`Of ${fmtInt(audit.billing_tagged_total)} bot-tagged Billing tickets, the text re-read keeps ${fmtInt(audit.billing_confirmed_billing)} as Billing and reclassifies ${fmtInt(audit.billing_reclassified)} (${nf1.format(num(audit.billing_reclassified_pct))}%), including ${fmtInt(audit.billing_reclassified_to_delivery)} to Delivery & Shipping. These are rule-based predictions, not verified truth.`);
  if (teams.length) {
    const byVol = [...teams].sort((a, b) => num(b.ticket_count) - num(a.ticket_count))[0];
    const byAgent = [...teams].filter((t) => !isNil(t.tickets_per_agent)).sort((a, b) => num(b.tickets_per_agent) - num(a.tickets_per_agent))[0];
    let s = `Largest team by raw volume: ${byVol.assigned_team} (${fmtInt(byVol.ticket_count)}, ${fmtPct(byVol.share_of_total)}).`;
    if (byAgent) s += ` Highest tickets per active agent: ${byAgent.assigned_team} (${fmtInt(byAgent.tickets_per_agent)}).`;
    out.push(s + " Raw volume alone is an incomplete headcount proxy — teams differ in staffing and handling time.");
  }
  if (num(bi.n_tickets_in_window) > 0) out.push(`Measured first-response breach rate: ${fmtPct(bi.overall_breach_rate)} (${fmtInt(bi.n_breached_in_window)} of ${fmtInt(bi.n_tickets_in_window)} usable helpdesk-era tickets). Estimated policy-based annual credit liability: ${fmtInr(bi.current_annual_liability_inr)} — an annualized estimate, not confirmed cash paid.`);
  if (!isNil(val.accuracy)) out.push(`Validation: ${fmtInt(val.n_correct)} of ${fmtInt(val.sample_size)} sampled tickets agree with AI-reviewed first-pass labels (${fmtPct(val.accuracy)}). This is not an independent human audit.`);
  if (num(ov.invalid_date_order) > 0) out.push(`${fmtInt(ov.invalid_date_order)} tickets have resolved_at earlier than created_at; they are flagged (not altered or deleted) and excluded from elapsed-time calculations.`);
  return out;
}
const esc0 = (v) => (isNil(v) ? "n/a" : String(v).slice(0, 10));

function renderOverview(d) {
  const ov = asObj(d.overview), bi = asObj(d.business_impact), audit = asObj(bi.category_tag_audit), val = asObj(d.validation);
  $("ov-kpis").innerHTML = [
    kpi("Total Tickets", fmtInt(ov.total_tickets), "in categorized_tickets.csv", "blue"),
    kpi("Categories Changed", fmtInt(ov.categories_changed), "predicted ≠ original bot category"),
    kpi("Category Change Rate", fmtPct(ov.category_change_rate), "rule-based, unverified"),
    kpi("Validation Accuracy", fmtPct(val.accuracy), `${fmtInt(val.n_correct)}/${fmtInt(val.sample_size)} · AI-reviewed, not a human audit`, "", B.ai),
    kpi("SLA Breach Rate", fmtPct(bi.overall_breach_rate), `${fmtInt(bi.n_breached_in_window)} of ${fmtInt(bi.n_tickets_in_window)} usable tickets`, "warn", B.meas),
    kpi("Est. Annual SLA Liability", fmtInr(bi.current_annual_liability_inr), "policy-based estimate, not cash paid", "warn", B.est),
    kpi("Active Agents", fmtInt(ov.active_agents), "current roster"),
    kpi("Billing Reclassified", fmtInt(audit.billing_reclassified), `${isNil(audit.billing_reclassified_pct) ? "—" : nf1.format(num(audit.billing_reclassified_pct)) + "%"} of ${fmtInt(audit.billing_tagged_total)} bot-tagged Billing`),
  ].join("");
  $("ov-notes").innerHTML = num(ov.invalid_date_order) > 0 ? `<div class="notice n-info"><b>${fmtInt(ov.invalid_date_order)}</b> tickets (legacy system) have <code>resolved_at</code> earlier than <code>created_at</code>. They are flagged, not altered or deleted, and excluded from elapsed-time calculations.</div>` : "";
  const f = buildFindings(d);
  $("ov-findings").innerHTML = f.length ? f.map((x) => `<li>${esc(x)}</li>`).join("") : "<li>No findings available yet.</li>";
  const src = asArray(d.monthly_team).length ? d.monthly_team : asArray(d.monthly_category);
  const tot = {}; asArray(src).forEach((r) => { const m = String(r?.month ?? ""); if (m) tot[m] = (tot[m] || 0) + num(r?.ticket_count); });
  const months = Object.keys(tot).sort();
  barChart("c-ov-volume", months, [{ label: "Tickets", data: months.map((m) => tot[m]), backgroundColor: "#2563eb" }]);
  lineChart("c-ov-cat", d.monthly_category, "predicted_category");
  lineChart("c-ov-team", d.monthly_team, "assigned_team");
}

function renderCategories(d) {
  const rows = asArray(d.category_summary);
  renderTable("t-cat", [
    { key: "category", label: "Category" },
    { key: "original_count", label: "Original (bot)", num: true, fmt: fmtInt },
    { key: "predicted_count", label: "Predicted", num: true, fmt: fmtInt },
    { key: "delta", label: "Δ", num: true, fmt: (v) => (isNil(v) ? "—" : (num(v) > 0 ? "+" : "") + fmtInt(v)) },
    { key: "share_of_total_predicted", label: "Share of predicted", num: true, fmt: fmtPct },
    { key: "original_reclassified_away", label: "Original tickets reclassified away", num: true, fmt: fmtInt },
    { key: "original_reclassified_away_rate", label: "% of original reclassified away", num: true, fmt: fmtPct },
  ], rows, { empty: "No category data." });
  doughnut("c-cat-dist", rows.map((r) => r.category), rows.map((r) => num(r.predicted_count)));
  barChart("c-cat-changes", rows.map((r) => r.category), [
    { label: "Original (bot)", data: rows.map((r) => num(r.original_count)), backgroundColor: "#94a3b8" },
    { label: "Predicted", data: rows.map((r) => num(r.predicted_count)), backgroundColor: "#2563eb" }]);
  lineChart("c-cat-monthly", d.monthly_category, "predicted_category");
  renderTable("t-cat-audit", [
    { key: "category", label: "Category" },
    { key: "bot_tagged_count", label: "Bot-tagged", num: true, fmt: fmtInt },
    { key: "reclassified_count", label: "Re-read", num: true, fmt: fmtInt },
    { key: "delta", label: "Δ", num: true, fmt: (v) => (isNil(v) ? "—" : (num(v) > 0 ? "+" : "") + fmtInt(v)) },
    { key: "pct_change", label: "% change", num: true, fmt: (v) => (isNil(v) ? "—" : (num(v) > 0 ? "+" : "") + nf1.format(num(v)) + "%") },
  ], asObj(asObj(d.business_impact).category_tag_audit).category_delta_table, { empty: "No category audit table in business_impact.json." });
  renderTable("t-cat-trans", [
    { key: "original_category", label: "Original" }, { key: "predicted_category", label: "Predicted" },
    { key: "ticket_count", label: "Tickets", num: true, fmt: fmtInt },
  ], d.category_transitions, { empty: "No reclassified tickets." });
}

function renderTeams(d) {
  renderTable("t-team", [
    { key: "assigned_team", label: "Team" },
    { key: "ticket_count", label: "Tickets", num: true, fmt: fmtInt },
    { key: "share_of_total", label: "Share", num: true, fmt: fmtPct },
    { key: "active_agents", label: "Active agents", num: true, fmt: fmtInt },
    { key: "tickets_per_agent", label: "Tickets / agent", num: true, fmt: fmtInt },
    { key: "median_handle_time_hours", label: "Median handle (h)", num: true, fmt: fmtNum1 },
    { key: "mean_handle_time_hours", label: "Mean handle (h)", num: true, fmt: fmtNum1 },
    { key: "transfer_rate", label: "Transfer rate", num: true, fmt: fmtPct },
    { key: "high_priority_share", label: "High-priority share", num: true, fmt: fmtPct },
    { key: "n_tickets_workload", label: "n (workload)", num: true, fmt: fmtInt },
  ], d.team_summary, { empty: "No team data." });
  const tv = asArray(d.team_volume).slice().sort((a, b) => num(b.ticket_count) - num(a.ticket_count));
  barChart("c-team-vol", tv.map((r) => r.assigned_team), [{ label: "Tickets", data: tv.map((r) => num(r.ticket_count)), backgroundColor: "#2563eb" }]);
  const tp = asArray(d.tickets_per_agent).slice().sort((a, b) => num(b.tickets_per_agent) - num(a.tickets_per_agent));
  barChart("c-team-per", tp.map((r) => r.assigned_team), [{ label: "Tickets per active agent", data: tp.map((r) => num(r.tickets_per_agent)), backgroundColor: "#f59e0b" }]);
  lineChart("c-team-monthly", d.monthly_team, "assigned_team");
}

function renderSLA(d) {
  const bi = asObj(d.business_impact), ch = asArray(bi.by_channel);
  $("sla-window").innerHTML = bi.data_window ? `<div class="notice n-info"><b>Data window:</b> ${esc(bi.data_window)}</div>` : "";
  $("sla-kpis-measured").innerHTML = [
    kpi("SLA Breach Rate", fmtPct(bi.overall_breach_rate), "first response past channel target", "warn", B.meas),
    kpi("Usable Tickets", fmtInt(bi.n_tickets_in_window), "helpdesk-era tickets in window", "", B.meas),
    kpi("Breached Tickets", fmtInt(bi.n_breached_in_window), "missed first-response target", "", B.meas),
    kpi("Policy Credit per Breach", fmtInr(bi.sla_breach_credit_inr), "per support policy", "", ""),
  ].join("");
  $("sla-kpis-est").innerHTML = [
    kpi("Annualized Tickets", fmtInt(bi.annualized_tickets), "window volume scaled to 365 days", "", B.est),
    kpi("Annualized Breaches", fmtInt(bi.annualized_breaches), "assumes the observed rate persists", "", B.est),
    kpi("Est. Annual Liability", fmtInr(bi.current_annual_liability_inr), "policy-based, not cash paid", "warn", B.est),
    kpi("Est. Quarterly Liability", fmtInr(bi.current_quarterly_liability_inr), "annual ÷ 4", "warn", B.est),
  ].join("");
  $("sla-est-note").innerHTML = `<div class="notice n-warn">${esc(bi.measured_vs_estimated || "Breach counts are measured from ticket timestamps; rupee figures are annualized policy-based estimates, not confirmed cash paid.")}</div>`;
  renderTable("t-sla", [
    { key: "channel", label: "Channel" },
    { key: "n_tickets", label: "Tickets", num: true, fmt: fmtInt },
    { key: "n_breached", label: "Breaches", num: true, fmt: fmtInt },
    { key: "breach_rate", label: "Breach rate", num: true, fmt: fmtPct },
    { key: "target_minutes", label: "Target (min)", num: true, fmt: fmtInt },
  ], ch, { empty: "No channel data in business_impact.json." });
  const labels = ch.map((r) => r.channel);
  barChart("c-sla-rate", labels, [{ label: "Breach rate", data: ch.map((r) => num(r.breach_rate)), backgroundColor: "#ef4444" }], { fmt: (v) => fmtPct(v) });
  barChart("c-sla-tix", labels, [{ label: "Tickets", data: ch.map((r) => num(r.n_tickets)), backgroundColor: "#2563eb" }]);
  barChart("c-sla-br", labels, [{ label: "Breached tickets", data: ch.map((r) => num(r.n_breached)), backgroundColor: "#f59e0b" }]);
  const sc = asArray(bi.scenarios);
  $("sla-scenario").innerHTML = (sc.length ? sc.map((s) => `
    <div class="notice n-warn"><b>Illustrative scenario — not a Vireo-stated target.</b> ${esc(s.label || "")}</div>
    <table><tbody>
      <tr><td>Current breach rate</td><td class="num">${fmtPct(bi.overall_breach_rate)}</td></tr>
      <tr><td>Scenario breach rate</td><td class="num">${fmtPct(s.target_rate)}</td></tr>
      <tr><td>Breaches avoided / year</td><td class="num">${fmtInt(s.breaches_avoided)}</td></tr>
      <tr><td>Credit avoided / year</td><td class="num">${fmtInr(s.annual_savings_inr)}</td></tr>
      <tr><td>Credit avoided / quarter</td><td class="num">${fmtInr(s.quarterly_savings_inr)}</td></tr></tbody></table>`).join("")
    : `<p class="small">No scenario found in business_impact.json.</p>`) + (bi.scenario_assumption ? `<p class="small">${esc(bi.scenario_assumption)}</p>` : "");
}

function renderBilling(d) {
  const bi = asObj(d.business_impact), a = asObj(bi.category_tag_audit);
  const tagged = num(a.billing_tagged_total), conf = num(a.billing_confirmed_billing), rec = num(a.billing_reclassified), del = num(a.billing_reclassified_to_delivery), oth = num(a.billing_reclassified_to_other_non_delivery);
  $("bill-kpis").innerHTML = [
    kpi("Bot-tagged Billing", fmtInt(tagged), "tickets carrying the Billing tag", "blue"),
    kpi("Confirmed Billing", fmtInt(conf), tagged ? fmtPct(conf / tagged) + " kept as Billing on re-read" : ""),
    kpi("Reclassified", fmtInt(rec), tagged ? fmtPct(rec / tagged) + " moved to another category" : "", "warn"),
    kpi("Reclassified to Delivery", fmtInt(del), tagged ? fmtPct(del / tagged) + " of Billing-tagged" : "", "warn"),
  ].join("");
  doughnut("c-bill-donut", ["Confirmed Billing", "Reclassified → Delivery & Shipping", "Reclassified → other categories"], [conf, del, oth], ["#10b981", "#f59e0b", "#94a3b8"]);
  const t = asArray(a.category_delta_table);
  barChart("c-bill-delta", t.map((r) => r.category), [
    { label: "Bot-tagged", data: t.map((r) => num(r.bot_tagged_count)), backgroundColor: "#94a3b8" },
    { label: "Re-read (predicted)", data: t.map((r) => num(r.reclassified_count)), backgroundColor: "#2563eb" }]);
  renderTable("t-bill", [{ key: "k", label: "Measure" }, { key: "v", label: "Tickets", num: true, fmt: fmtInt }, { key: "p", label: "% of Billing-tagged", num: true, fmt: fmtPct }], [
    { k: "Bot-tagged Billing", v: tagged, p: tagged ? 1 : null },
    { k: "Confirmed Billing (re-read agrees)", v: conf, p: tagged ? conf / tagged : null },
    { k: "Reclassified to another category", v: rec, p: tagged ? rec / tagged : null },
    { k: "…of which Delivery & Shipping", v: del, p: tagged ? del / tagged : null },
    { k: "…of which other categories", v: oth, p: tagged ? oth / tagged : null },
  ], { empty: "No billing audit data." });
  renderTable("t-bill-break", [{ key: "predicted_category", label: "Predicted category" }, { key: "ticket_count", label: "Tickets", num: true, fmt: fmtInt }, { key: "share", label: "Share", num: true, fmt: fmtPct }], d.billing_breakdown, { empty: "No Billing-tagged tickets found." });
}

function renderValidation(d) {
  const v = asObj(d.validation);
  $("val-kpis").innerHTML = [
    kpi("Sample Size", fmtInt(v.sample_size), isNil(v.random_seed) ? "" : "random seed " + v.random_seed),
    kpi("Agreement (accuracy)", fmtPct(v.accuracy), "with AI-reviewed labels", "blue", B.ai),
    kpi("Correct", fmtInt(v.n_correct), ""), kpi("Errors", fmtInt(v.n_errors), "", "warn"),
    kpi("Error Rate", fmtPct(v.error_rate), ""),
    kpi("Model", v.model || "—", "deterministic rules, no paid API"),
    kpi("Taxonomy Version", v.taxonomy_version || "—", ""),
  ].join("");
  $("val-meta").innerHTML = (v.reviewer_type ? `<div class="notice n-info"><b>Reviewer:</b> ${esc(v.reviewer_type)}</div>` : "") + (v.method ? `<p class="small" style="margin:0 0 14px"><b>Method:</b> ${esc(v.method)}</p>` : "");
  const pc = asArray(v.per_category_precision).slice().sort((a, b) => num(a.precision) - num(b.precision));
  barChart("c-val-prec", pc.map((r) => `${r.category} (n=${fmtInt(r.n_predicted)})`), [{ label: "Sample precision", data: pc.map((r) => num(r.precision)), backgroundColor: pc.map((r) => (num(r.precision) >= 0.9 ? "#10b981" : num(r.precision) >= 0.75 ? "#f59e0b" : "#ef4444")) }], { horizontal: true, max: 1, fmt: (x) => fmtPct(x, 0) });
  renderTable("t-val-prec", [{ key: "category", label: "Category" }, { key: "n_predicted", label: "Predicted in sample", num: true, fmt: fmtInt }, { key: "n_errors", label: "Errors", num: true, fmt: fmtInt }, { key: "precision", label: "Precision", num: true, fmt: fmtPct }], pc.slice().reverse(), { empty: "No per-category data." });
  renderTable("t-val-fm", [{ key: "mode", label: "Failure mode" }, { key: "count", label: "Errors", num: true, fmt: fmtInt }], v.failure_modes, { empty: "No failure modes reported." });
  renderTable("t-val-err", [
    { key: "ticket_id", label: "Ticket" }, { key: "predicted_category", label: "Predicted" }, { key: "true_category", label: "AI-reviewed label" },
    { key: "bot_original_category", label: "Bot category" }, { key: "failure_mode", label: "Failure mode" }], v.errors_detail, { empty: "No validation errors reported." });
  $("val-limit").textContent = v.limitation || "AI-assisted first-pass validation — not an independent human audit.";
}

/* ---------- explorer ---------- */
function fillSelect(id, values, label) {
  const el = $(id); if (!el) return; const cur = el.value;
  el.innerHTML = `<option value="">All</option>` + asArray(values).map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
  if ([...el.options].some((o) => o.value === cur)) el.value = cur;
}
function initExplorerFilters(d) {
  const fo = asObj(d.filter_options);
  fillSelect("f-original", fo.original_category); fillSelect("f-predicted", fo.predicted_category); fillSelect("f-team", fo.team);
  fillSelect("f-channel", fo.channel); fillSelect("f-priority", fo.priority); fillSelect("f-source", fo.source_system);
}
const changedBadge = (v) => (v === true || v === "true" ? '<span class="badge b-red">Changed</span>' : '<span class="badge b-gray">Same</span>');
const invalidBadge = (v) => (v === true || v === "true" ? '<span class="badge b-red">Invalid</span>' : '<span class="badge b-gray">OK</span>');
const EX_COLS = [
  { key: "ticket_id", label: "Ticket ID", sort: "ticket_id" },
  { key: "created_at", label: "Created", sort: "created_at" },
  { key: "channel", label: "Channel", sort: "channel" },
  { key: "category", label: "Original Category", sort: "category" },
  { key: "predicted_category", label: "Predicted Category", sort: "predicted_category" },
  { key: "category_changed", label: "Changed", sort: "category_changed", html: changedBadge },
  { key: "priority", label: "Priority", sort: "priority" },
  { key: "assigned_team", label: "Assigned Team", sort: "assigned_team" },
  { key: "agent_id", label: "Agent", sort: "agent_id" },
  { key: "transfers", label: "Transfers", sort: "transfers", num: true, fmt: fmtInt },
  { key: "csat_score", label: "CSAT", sort: "csat_score", num: true, fmt: (v) => (isNil(v) ? "—" : nf1.format(num(v))) },
  { key: "source_system", label: "Source System", sort: "source_system" },
  { key: "invalid_date_order", label: "Invalid Date Order", sort: "invalid_date_order", html: invalidBadge },
];
function explorerParams() {
  const p = new URLSearchParams();
  const add = (k, id) => { const v = $(id).value.trim(); if (v) p.set(k, v); };
  add("search", "f-search"); add("original_category", "f-original"); add("predicted_category", "f-predicted"); add("team", "f-team");
  add("channel", "f-channel"); add("priority", "f-priority"); add("changed", "f-changed"); add("source_system", "f-source");
  p.set("page", S.ex.page); p.set("page_size", $("f-size").value); p.set("sort_by", S.ex.sort_by); p.set("sort_dir", S.ex.sort_dir);
  return p;
}
async function loadExplorer() {
  try {
    const r = await api("/api/tickets?" + explorerParams().toString());
    const rows = asArray(r.rows); S.ex.page = num(r.page, 1);
    renderTable("ex-table", EX_COLS, rows, { sortable: true, sortBy: S.ex.sort_by, sortDir: S.ex.sort_dir, empty: "No tickets match the current filters.",
      onSort: (col) => { if (S.ex.sort_by === col) S.ex.sort_dir = S.ex.sort_dir === "asc" ? "desc" : "asc"; else { S.ex.sort_by = col; S.ex.sort_dir = "asc"; } S.ex.page = 1; loadExplorer(); } });
    const size = num(r.page_size, 50), total = num(r.total);
    $("ex-info").textContent = total ? `Showing ${fmtInt((S.ex.page - 1) * size + 1)}–${fmtInt(Math.min(S.ex.page * size, total))} of ${fmtInt(total)} tickets` : "0 tickets";
    $("ex-page").textContent = `Page ${fmtInt(S.ex.page)} of ${fmtInt(num(r.pages, 1))}`;
    $("ex-prev").disabled = S.ex.page <= 1; $("ex-next").disabled = S.ex.page >= num(r.pages, 1);
    showBanner("", "");
  } catch (e) { showBanner("n-err", "Ticket Explorer: " + e.message); }
}
function initExplorer() {
  if (S.exReady) return; S.exReady = true;
  let t; const deb = () => { clearTimeout(t); t = setTimeout(() => { S.ex.page = 1; loadExplorer(); }, 300); };
  $("f-search").addEventListener("input", deb);
  ["f-original", "f-predicted", "f-team", "f-channel", "f-priority", "f-changed", "f-source", "f-size"].forEach((id) => $(id).addEventListener("change", () => { S.ex.page = 1; loadExplorer(); }));
  $("f-reset").addEventListener("click", () => { $("f-search").value = ""; ["f-original", "f-predicted", "f-team", "f-channel", "f-priority", "f-changed", "f-source"].forEach((id) => ($(id).value = "")); S.ex.page = 1; loadExplorer(); });
  $("ex-prev").addEventListener("click", () => { if (S.ex.page > 1) { S.ex.page--; loadExplorer(); } });
  $("ex-next").addEventListener("click", () => { S.ex.page++; loadExplorer(); });
}

/* ---------- orchestration ---------- */
const SECTIONS = { overview: renderOverview, categories: renderCategories, teams: renderTeams, sla: renderSLA, billing: renderBilling, validation: renderValidation };
function renderAll() {
  const d = asObj(S.data);
  $("lastRefresh").textContent = fmtDate(asObj(d.metadata).last_refresh);
  Object.entries(SECTIONS).forEach(([name, fn]) => {
    try { fn(d); } catch (e) { console.error("Render failed for", name, e); const pg = $("page-" + name); if (pg) pg.insertAdjacentHTML("afterbegin", `<div class="notice n-err">Could not render this section: ${esc(e.message)}</div>`); }
  });
  initExplorerFilters(d);
}
async function loadDashboard() {
  setLoading(true);
  try {
    S.data = await api("/api/dashboard");
    document.querySelectorAll(".page > .notice.n-err").forEach((n) => n.remove());
    renderAll(); showBanner("", "");
    if (S.exReady) loadExplorer();
    return true;
  } catch (e) { showBanner("n-err", "Unable to load dashboard data: " + e.message); return false; }
  finally { setLoading(false); }
}
function showPage(name) {
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + name));
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.page === name));
  $("sidebar").classList.remove("open");
  if (name === "explorer") { initExplorer(); loadExplorer(); }
  Object.values(S.charts).forEach((c) => { try { c.resize(); } catch (e) {} });
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  window.scrollTo(0, 0);
}
async function pollRefresh() {
  try {
    const s = await api("/api/refresh/status");
    if (s.running) { $("refreshBtn").textContent = "Refreshing..."; S.polling = setTimeout(pollRefresh, 2000); return; }
    $("refreshBtn").disabled = false; $("refreshBtn").textContent = "Refresh Analysis";
    if (s.state === "error") { showBanner("n-err", s.message || "Refresh failed."); setLoading(false); return; }
    const ok = await loadDashboard();
    if (ok) { showBanner("n-ok", s.message || "Refresh complete."); setTimeout(() => { if ($("errbanner").querySelector(".n-ok")) showBanner("", ""); }, 6000); }
  } catch (e) { $("refreshBtn").disabled = false; $("refreshBtn").textContent = "Refresh Analysis"; setLoading(false); showBanner("n-err", "Refresh status check failed: " + e.message); }
}
async function startRefresh() {
  $("refreshBtn").disabled = true; $("refreshBtn").textContent = "Refreshing..."; setLoading(true); showBanner("n-info", "Running the analysis pipeline — this can take a little while…");
  try { await api("/api/refresh", { method: "POST" }); clearTimeout(S.polling); S.polling = setTimeout(pollRefresh, 2000); }
  catch (e) { $("refreshBtn").disabled = false; $("refreshBtn").textContent = "Refresh Analysis"; setLoading(false); showBanner("n-err", "Could not start refresh: " + e.message); }
}
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("#nav button").forEach((b) => b.addEventListener("click", () => showPage(b.dataset.page)));
  $("burger").addEventListener("click", () => $("sidebar").classList.toggle("open"));
  $("refreshBtn").addEventListener("click", startRefresh);
  window.addEventListener("resize", () => Object.values(S.charts).forEach((c) => { try { c.resize(); } catch (e) {} }));
  const start = (location.hash || "").replace("#", "");
  loadDashboard().then(() => {
    // if a refresh is already running (e.g. page reloaded mid-run), resume polling
    api("/api/refresh/status").then((s) => { if (s.running) { $("refreshBtn").disabled = true; pollRefresh(); } }).catch(() => {});
  });
  if (start && document.getElementById("page-" + start)) showPage(start);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)