"""
Step 4 – Create a static-evals analysis dashboard in Langfuse.

The Langfuse dashboard API is only accessible via tRPC (session auth), not via
the public API key auth used by the SDK.  This script therefore:

  1. Signs into Langfuse using your email + password to get a session cookie.
  2. Resolves the project ID from the API keys stored in the environment.
  3. Creates a set of analysis widgets (score averages, score distributions).
  4. Creates a new Dashboard and arranges the widgets on it.

Usage
-----
  python step4_create_dashboard.py \\
      --email admin@example.com \\
      --password yourpassword \\
      --dashboard-name "Static Evals Analysis"

  # You can also set credentials via env vars to avoid typing them:
  LANGFUSE_ADMIN_EMAIL=... LANGFUSE_ADMIN_PASSWORD=... \\
      python step4_create_dashboard.py
"""

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path

import requests
import yaml


# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "langfuse_static_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── tRPC HTTP helpers ─────────────────────────────────────────────────────────

def trpc_mutation(session: requests.Session, base_url: str, procedure: str, data: dict) -> dict:
    """Call a tRPC mutation via HTTP POST."""
    url = f"{base_url}/api/trpc/{procedure}"
    resp = session.post(url, json={"json": data})
    resp.raise_for_status()
    result = resp.json()
    return result["result"]["data"]["json"]


def trpc_query(session: requests.Session, base_url: str, procedure: str, data: dict) -> dict:
    """Call a tRPC query via HTTP GET."""
    encoded_input = urllib.parse.quote(json.dumps({"json": data}))
    url = f"{base_url}/api/trpc/{procedure}?input={encoded_input}"
    resp = session.get(url)
    resp.raise_for_status()
    result = resp.json()
    return result["result"]["data"]["json"]


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(base_url: str, email: str, password: str) -> requests.Session:
    """Authenticate with NextAuth credentials and return a session with cookies."""
    s = requests.Session()

    # Step 1: get CSRF token
    csrf_resp = s.get(f"{base_url}/api/auth/csrf")
    csrf_resp.raise_for_status()
    csrf_token = csrf_resp.json()["csrfToken"]

    # Step 2: post credentials
    login_resp = s.post(
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

    # Verify we have a session cookie
    if "next-auth.session-token" not in s.cookies and \
       "__Secure-next-auth.session-token" not in s.cookies:
        sys.exit(
            "[step4] Login failed – no session cookie received. "
            "Check your email/password and that the Langfuse server is running."
        )

    print(f"[step4] Logged in as '{email}'")
    return s


def get_project_id(base_url: str, public_key: str, secret_key: str) -> str:
    """Fetch the project ID from /api/public/projects using Basic auth."""
    resp = requests.get(
        f"{base_url}/api/public/projects",
        auth=(public_key, secret_key),
    )
    resp.raise_for_status()
    projects = resp.json().get("data", [])
    if not projects:
        sys.exit("[step4] No projects returned by /api/public/projects.")
    # Use the first project (the API key already scopes us to the right one)
    proj = projects[0]
    print(f"[step4] Using project: '{proj['name']}' ({proj['id']})")
    return proj["id"]


# ── Widget definitions ────────────────────────────────────────────────────────
#
# Score names as stored by step3_run_eval.py:
#   Rubric  → "rubric_eval.resolution", "rubric_eval.micro_skills",
#             "rubric_eval.context", "rubric_eval.language"
#   Unit tests → "empathy_chaining", "forbidden", "multiple_tasks",
#                "pacing", "questions", "risk_assessment"

RUBRIC_SCORE_NAMES = [
    "rubric_eval.resolution",
    "rubric_eval.micro_skills",
    "rubric_eval.context",
    "rubric_eval.language",
]

UNIT_TEST_SCORE_NAMES = [
    "empathy_chaining",
    "forbidden",
    "multiple_tasks",
    "pacing",
    "questions",
    "risk_assessment",
]

UNIT_TEST_LABELS = {
    "empathy_chaining": "Empathy Chaining",
    "forbidden":        "Forbidden Statements",
    "multiple_tasks":   "Multiple Tasks",
    "pacing":           "Pacing",
    "questions":        "Questions",
    "risk_assessment":  "Risk Assessment",
}


def _name_filter(score_name: str) -> dict:
    """Exact-match filter on score.name."""
    return {"column": "name", "operator": "=", "value": score_name, "type": "string"}


def _name_options_filter(score_names: list[str]) -> dict:
    """Multi-value filter on score.name."""
    return {"column": "name", "operator": "any of", "value": score_names, "type": "stringOptions"}


def widget_definitions(project_id: str) -> list[dict]:
    """Return the list of widgets to create for the static evals dashboard."""

    def base(name, description, view, dimensions, metrics, filters, chart_type, chart_config=None):
        w = {
            "projectId": project_id,
            "name": name,
            "description": description,
            "view": view,
            "dimensions": dimensions,
            "metrics": metrics,
            "filters": filters,
            "chartType": chart_type,
            "chartConfig": chart_config or {"type": chart_type},
        }
        return w

    widgets = []

    # 1. Avg score per dimension – all scores overview
    widgets.append(base(
        name="Avg Score – All Dimensions",
        description="Mean score across every eval dimension (rubric + unit tests)",
        view="scores-numeric",
        dimensions=[{"field": "name"}],
        metrics=[{"measure": "value", "agg": "avg"}],
        filters=[_name_options_filter(RUBRIC_SCORE_NAMES + UNIT_TEST_SCORE_NAMES)],
        chart_type="HORIZONTAL_BAR",
    ))

    # 2. Rubric: avg per feature (resolution / micro_skills / context / language)
    widgets.append(base(
        name="Rubric: Avg Score per Feature",
        description="Average rubric score for each of the 4 evaluation dimensions",
        view="scores-numeric",
        dimensions=[{"field": "name"}],
        metrics=[{"measure": "value", "agg": "avg"}],
        filters=[_name_options_filter(RUBRIC_SCORE_NAMES)],
        chart_type="HORIZONTAL_BAR",
    ))

    # 3–8. One histogram per unit-test eval prompt
    for score_name, label in UNIT_TEST_LABELS.items():
        widgets.append(base(
            name=f"Unit Test: {label}",
            description=f"Score distribution for '{score_name}' across all evaluated traces",
            view="scores-numeric",
            dimensions=[],
            metrics=[{"measure": "value", "agg": "histogram"}],
            filters=[_name_filter(score_name)],
            chart_type="HISTOGRAM",
            chart_config={"type": "HISTOGRAM"},
        ))

    # 9. Total generation traces
    widgets.append(base(
        name="Total Traces (Generations)",
        description="Total number of copilot generation traces in this experiment",
        view="traces",
        dimensions=[],
        metrics=[{"measure": "count", "agg": "count"}],
        filters=[],
        chart_type="NUMBER",
    ))

    return widgets


# ── Layout helpers ────────────────────────────────────────────────────────────

def build_layout(widget_ids: list[str]) -> dict:
    """Arrange widget IDs in a grid layout.

    Row 0  (full):  Avg all dims
    Row 4  (full):  Rubric avg
    Rows 8-12: 3-column grid for 6 unit-test histograms
    Row 16 (left quarter): Total traces number
    """
    positions = [
        # Row 1 – overview bar
        (0, 0, 12, 4),
        # Row 2 – rubric avg bar
        (0, 4, 12, 4),
        # Row 3 – unit tests row 1 (3 cols)
        (0, 8,  4, 4),
        (4, 8,  4, 4),
        (8, 8,  4, 4),
        # Row 4 – unit tests row 2 (3 cols)
        (0, 12, 4, 4),
        (4, 12, 4, 4),
        (8, 12, 4, 4),
        # Row 5 – total traces number
        (0, 16, 3, 2),
    ]
    widgets = []
    for i, (widget_id, (x, y, x_size, y_size)) in enumerate(
        zip(widget_ids, positions)
    ):
        widgets.append({
            "type": "widget",
            "id": f"layout-{i}",
            "widgetId": widget_id,
            "x": x,
            "y": y,
            "x_size": x_size,
            "y_size": y_size,
        })
    return {"widgets": widgets}


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Create static-evals dashboard in Langfuse")
    p.add_argument("--email",
                   default=os.environ.get("LANGFUSE_ADMIN_EMAIL", ""),
                   help="Langfuse login email (or set LANGFUSE_ADMIN_EMAIL)")
    p.add_argument("--password",
                   default=os.environ.get("LANGFUSE_ADMIN_PASSWORD", ""),
                   help="Langfuse login password (or set LANGFUSE_ADMIN_PASSWORD)")
    p.add_argument("--base-url",
                   default=os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000"))
    p.add_argument("--dashboard-name", default="Static Evals Analysis")
    p.add_argument("--dashboard-description",
                   default="Automated dashboard for copilot static evaluation results")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.email or not args.password:
        sys.exit(
            "[step4] Email and password are required.\n"
            "  Pass --email / --password, or set LANGFUSE_ADMIN_EMAIL / LANGFUSE_ADMIN_PASSWORD.\n\n"
            "  Alternatively, create the dashboard manually in the Langfuse UI:\n"
            f"  {args.base_url}/dashboard\n"
        )

    # ── Log in ────────────────────────────────────────────────────────────────
    session = login(args.base_url, args.email, args.password)

    # ── Resolve project ID ────────────────────────────────────────────────────
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    project_id = get_project_id(args.base_url, public_key, secret_key)

    # ── Create widgets ────────────────────────────────────────────────────────
    widget_defs = widget_definitions(project_id)
    widget_ids = []
    print(f"[step4] Creating {len(widget_defs)} widgets …")
    for wdef in widget_defs:
        try:
            result = trpc_mutation(session, args.base_url, "dashboardWidgets.create", wdef)
            widget_id = result["widget"]["id"]
            widget_ids.append(widget_id)
            print(f"  [ok] '{wdef['name']}' → {widget_id}")
        except Exception as e:
            print(f"  [WARN] Widget '{wdef['name']}' failed: {e}")

    if not widget_ids:
        sys.exit("[step4] No widgets were created. Dashboard creation aborted.")

    # ── Create dashboard ──────────────────────────────────────────────────────
    print(f"[step4] Creating dashboard '{args.dashboard_name}' …")
    try:
        dash_result = trpc_mutation(
            session, args.base_url, "dashboard.createDashboard",
            {
                "projectId": project_id,
                "name": args.dashboard_name,
                "description": args.dashboard_description,
            }
        )
        dashboard_id = dash_result["id"]
        print(f"[step4] Dashboard created: {dashboard_id}")
    except Exception as e:
        sys.exit(f"[step4] Failed to create dashboard: {e}")

    # ── Add widgets to dashboard ──────────────────────────────────────────────
    print(f"[step4] Adding {len(widget_ids)} widgets to dashboard …")
    layout = build_layout(widget_ids)
    try:
        trpc_mutation(
            session, args.base_url, "dashboard.updateDashboardDefinition",
            {
                "projectId": project_id,
                "dashboardId": dashboard_id,
                "definition": layout,
            }
        )
        print(f"[step4] Layout applied.")
    except Exception as e:
        print(f"[step4] Warning: could not apply layout: {e}")
        print(f"[step4] Widgets were created – you can arrange them manually.")

    print(
        f"\n[step4] ✅ Dashboard ready!\n"
        f"  View it at: {args.base_url}/project/{project_id}/dashboards/{dashboard_id}"
    )


if __name__ == "__main__":
    main()
