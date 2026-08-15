"""
Step 3 – Create a dynamic-evals analysis dashboard in Langfuse.

Uses the same tRPC-based approach as the static evals dashboard:
  1. Signs in with email + password to get a session cookie.
  2. Resolves the project ID from the API keys stored in the environment.
  3. Creates widgets for all dynamic eval score dimensions.
  4. Creates a new Dashboard and arranges the widgets on it.

Score dimensions produced by step2_run_eval.py:
  Language Patterns:
    language_patterns.grammar_and_punctuation
    language_patterns.message_length_and_complexity
    language_patterns.authenticity_and_natural_flow
  Presenting Concern:
    presenting_concern.score
  Big Five Traits (simple):
    big_five_traits_simple.score
  Emotional Realism:
    emotional_realism.cognitive_distortions
    emotional_realism.physical_or_behavioral_symptoms_of_distress
    emotional_realism.depth_of_emotion_presenting_issue
    emotional_realism.authentic_progression_of_emotional_states

Usage
-----
  python step3_create_dashboard.py \\
      --email admin@example.com --password yourpassword

  LANGFUSE_ADMIN_EMAIL=... LANGFUSE_ADMIN_PASSWORD=... \\
      python step3_create_dashboard.py --dashboard-name "Dynamic Evals Analysis"
"""

import argparse
import json
import os
import sys
import urllib.parse
import uuid
from pathlib import Path

import requests
import yaml


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "langfuse_dynamic_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── tRPC HTTP helpers ─────────────────────────────────────────────────────────

def trpc_mutation(session: requests.Session, base_url: str, procedure: str, data: dict) -> dict:
    url = f"{base_url}/api/trpc/{procedure}"
    resp = session.post(url, json={"json": data})
    resp.raise_for_status()
    return resp.json()["result"]["data"]["json"]


def trpc_query(session: requests.Session, base_url: str, procedure: str, data: dict) -> dict:
    encoded = urllib.parse.quote(json.dumps({"json": data}))
    url = f"{base_url}/api/trpc/{procedure}?input={encoded}"
    resp = session.get(url)
    resp.raise_for_status()
    return resp.json()["result"]["data"]["json"]


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(base_url: str, email: str, password: str) -> requests.Session:
    s = requests.Session()
    csrf_resp = s.get(f"{base_url}/api/auth/csrf")
    csrf_resp.raise_for_status()
    csrf_token = csrf_resp.json()["csrfToken"]

    s.post(
        f"{base_url}/api/auth/callback/credentials",
        data={
            "csrfToken": csrf_token,
            "email": email,
            "password": password,
            "callbackUrl": base_url,
            "json": "true",
        },
        allow_redirects=True,
    )

    if ("next-auth.session-token" not in s.cookies and
            "__Secure-next-auth.session-token" not in s.cookies):
        sys.exit(
            "[step3] Login failed – no session cookie received. "
            "Check your email/password and that the Langfuse server is running."
        )
    print(f"[step3] Logged in as '{email}'")
    return s


def get_project_id(base_url: str, public_key: str, secret_key: str) -> str:
    resp = requests.get(
        f"{base_url}/api/public/projects",
        auth=(public_key, secret_key),
    )
    resp.raise_for_status()
    projects = resp.json().get("data", [])
    if not projects:
        sys.exit("[step3] No projects returned by /api/public/projects.")
    proj = projects[0]
    print(f"[step3] Using project: '{proj['name']}' ({proj['id']})")
    return proj["id"]


# ── Score name constants ───────────────────────────────────────────────────────

LANGUAGE_SCORES = [
    "language_patterns.grammar_and_punctuation",
    "language_patterns.message_length_and_complexity",
    "language_patterns.authenticity_and_natural_flow",
]

PRESENTING_CONCERN_SCORES = [
    "presenting_concern.score",
]

BIG_FIVE_SCORES = [
    "big_five_traits_simple.score",
]

EMOTIONAL_REALISM_SCORES = [
    "emotional_realism.cognitive_distortions",
    "emotional_realism.physical_or_behavioral_symptoms_of_distress",
    "emotional_realism.depth_of_emotion_presenting_issue",
    "emotional_realism.authentic_progression_of_emotional_states",
]

ALL_SCORES = (
    LANGUAGE_SCORES
    + PRESENTING_CONCERN_SCORES
    + BIG_FIVE_SCORES
    + EMOTIONAL_REALISM_SCORES
)

SCORE_LABELS = {
    "language_patterns.grammar_and_punctuation":         "Grammar & Punctuation",
    "language_patterns.message_length_and_complexity":   "Message Length & Complexity",
    "language_patterns.authenticity_and_natural_flow":   "Authenticity & Natural Flow",
    "presenting_concern.score":                          "Presenting Concern",
    "big_five_traits_simple.score":                      "Big Five Traits",
    "emotional_realism.cognitive_distortions":           "Cognitive Distortions",
    "emotional_realism.physical_or_behavioral_symptoms_of_distress": "Physical/Behavioral Symptoms",
    "emotional_realism.depth_of_emotion_presenting_issue": "Depth of Emotion",
    "emotional_realism.authentic_progression_of_emotional_states": "Authentic Progression",
}


def _name_filter(score_name: str) -> dict:
    return {"column": "name", "operator": "=", "value": score_name, "type": "string"}


def _name_options_filter(score_names: list) -> dict:
    return {"column": "name", "operator": "any of", "value": score_names, "type": "stringOptions"}


# ── Widget definitions ────────────────────────────────────────────────────────

def widget_definitions(project_id: str) -> list[dict]:

    def base(name, description, view, dimensions, metrics, filters, chart_type, chart_config=None):
        return {
            "projectId":   project_id,
            "name":        name,
            "description": description,
            "view":        view,
            "dimensions":  dimensions,
            "metrics":     metrics,
            "filters":     filters,
            "chartType":   chart_type,
            "chartConfig": chart_config or {"type": chart_type},
        }

    widgets = []

    # 1. All-dimensions overview bar chart
    widgets.append(base(
        name="Avg Score – All Dimensions",
        description="Mean score across every dynamic eval dimension",
        view="scores-numeric",
        dimensions=[{"field": "name"}],
        metrics=[{"measure": "value", "agg": "avg"}],
        filters=[_name_options_filter(ALL_SCORES)],
        chart_type="HORIZONTAL_BAR",
    ))

    # 2. Language Patterns – avg per sub-metric
    widgets.append(base(
        name="Language Patterns – Avg per Metric",
        description="Average score for each language pattern sub-dimension",
        view="scores-numeric",
        dimensions=[{"field": "name"}],
        metrics=[{"measure": "value", "agg": "avg"}],
        filters=[_name_options_filter(LANGUAGE_SCORES)],
        chart_type="HORIZONTAL_BAR",
    ))

    # 3. Emotional Realism – avg per sub-metric
    widgets.append(base(
        name="Emotional Realism – Avg per Metric",
        description="Average score for each emotional realism sub-dimension",
        view="scores-numeric",
        dimensions=[{"field": "name"}],
        metrics=[{"measure": "value", "agg": "avg"}],
        filters=[_name_options_filter(EMOTIONAL_REALISM_SCORES)],
        chart_type="HORIZONTAL_BAR",
    ))

    # 4–12. One histogram per score dimension
    for score_name, label in SCORE_LABELS.items():
        widgets.append(base(
            name=f"Distribution: {label}",
            description=f"Score distribution for '{score_name}' across all evaluated dialogues",
            view="scores-numeric",
            dimensions=[],
            metrics=[{"measure": "value", "agg": "histogram"}],
            filters=[_name_filter(score_name)],
            chart_type="HISTOGRAM",
            chart_config={"type": "HISTOGRAM"},
        ))

    # Last: total traces
    widgets.append(base(
        name="Total Eval Traces",
        description="Total number of dialogue evaluation traces in this experiment",
        view="traces",
        dimensions=[],
        metrics=[{"measure": "count", "agg": "count"}],
        filters=[],
        chart_type="NUMBER",
    ))

    return widgets


# ── Layout ────────────────────────────────────────────────────────────────────

def build_definition(widget_ids: list) -> dict:
    """
    Build a DashboardDefinitionSchema dict for updateDashboardDefinition.
    Each tile: { type, id (placement uuid), widgetId, x, y, x_size, y_size }

    Row 0  (full-width):    All-dimensions overview bar
    Row 4  (half + half):   Language Patterns bar | Emotional Realism bar
    Row 8  (3-col grid):    First 3 histograms
    Row 12 (3-col grid):    Next 3 histograms
    Row 16 (3-col grid):    Next 3 histograms
    Row 20 (quarter):       Total traces number
    """
    positions = [
        # All-dims bar – full width
        (0,  0,  12, 4),
        # Language + Emotional side-by-side
        (0,  4,  6,  4),
        (6,  4,  6,  4),
    ]

    # Histograms for each score – 3 per row
    n_histo = len(SCORE_LABELS)
    for idx in range(n_histo):
        col = (idx % 3) * 4
        row = 8 + (idx // 3) * 4
        positions.append((col, row, 4, 4))

    # Total traces number tile
    end_row = 8 + ((n_histo + 2) // 3) * 4
    positions.append((0, end_row, 3, 4))

    widgets = []
    for wid, (x, y, w, h) in zip(widget_ids, positions):
        widgets.append({
            "type":     "widget",
            "id":       str(uuid.uuid4()),   # unique placement ID
            "widgetId": wid,
            "x":        x,
            "y":        y,
            "x_size":   w,
            "y_size":   h,
        })
    return {"widgets": widgets}


# ── Orchestration ─────────────────────────────────────────────────────────────

def create_dashboard(email: str, password: str, dashboard_name: str) -> None:
    base_url   = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000").rstrip("/")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        sys.exit("[step3] ❌ LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set.")

    session    = login(base_url, email, password)
    project_id = get_project_id(base_url, public_key, secret_key)

    # ── Create widgets ──────────────────────────────────────────────────────
    # Procedure path: dashboardWidgets.create  (router mounted as
    # 'dashboardWidgets' in root.ts, procedure named 'create')
    widget_defs = widget_definitions(project_id)
    widget_ids  = []
    print(f"\n[step3] Creating {len(widget_defs)} widget(s) …")
    for wdef in widget_defs:
        try:
            result = trpc_mutation(session, base_url, "dashboardWidgets.create", wdef)
            # Returns { success: true, widget: { id: ..., ... } }
            wid = result.get("widget", {}).get("id")
            widget_ids.append(wid)
            print(f"  ✓ '{wdef['name']}' (id={wid})")
        except Exception as exc:
            print(f"  ✗ '{wdef['name']}': {exc}")
            widget_ids.append(None)

    valid_widget_ids = [wid for wid in widget_ids if wid]
    if not valid_widget_ids:
        sys.exit("[step3] No widgets were created successfully.")

    # ── Create empty dashboard ───────────────────────────────────────────────
    # Procedure path: dashboard.createDashboard
    print(f"\n[step3] Creating dashboard '{dashboard_name}' …")
    try:
        result = trpc_mutation(session, base_url, "dashboard.createDashboard", {
            "projectId":   project_id,
            "name":        dashboard_name,
            "description": "Dynamic eval scores across patient profile dimensions",
        })
        dashboard_id = result.get("id", "?")
        print(f"[step3] Dashboard created: id={dashboard_id}")
    except Exception as exc:
        sys.exit(f"[step3] ✗ Could not create dashboard: {exc}")

    # ── Apply widget layout via updateDashboardDefinition ────────────────────
    # Procedure path: dashboard.updateDashboardDefinition
    # Definition schema: { widgets: [{ type, id (placement), widgetId, x, y, x_size, y_size }] }
    definition = build_definition(valid_widget_ids)
    try:
        trpc_mutation(session, base_url, "dashboard.updateDashboardDefinition", {
            "projectId":   project_id,
            "dashboardId": dashboard_id,
            "definition":  definition,
        })
        print(f"[step3] ✅ Dashboard layout applied ({len(valid_widget_ids)} widgets).")
        print(f"         View at: {base_url}/project/{project_id}/dashboards/{dashboard_id}")
    except Exception as exc:
        print(f"[step3] ⚠️  Could not apply layout (dashboard exists but has no tiles): {exc}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Create dynamic-evals analysis dashboard in Langfuse")
    p.add_argument("--email",          default=os.environ.get("LANGFUSE_ADMIN_EMAIL", ""),
                   help="Langfuse admin email")
    p.add_argument("--password",       default=os.environ.get("LANGFUSE_ADMIN_PASSWORD", ""),
                   help="Langfuse admin password")
    p.add_argument("--dashboard-name", default="Dynamic Evals Analysis",
                   help="Name for the new dashboard")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.email or not args.password:
        sys.exit(
            "[step3] --email and --password are required (or set "
            "LANGFUSE_ADMIN_EMAIL / LANGFUSE_ADMIN_PASSWORD env vars)."
        )
    create_dashboard(args.email, args.password, args.dashboard_name)


if __name__ == "__main__":
    main()
