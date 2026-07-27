"""End-to-end smoke tests for the Regression Playground Streamlit app."""

import json
from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
sys.path.insert(0, str(APP_PATH.parent))

import regression_utils as ru  # noqa: E402
import theme  # noqa: E402


WORKSPACES = [
    "Fit the Line",
    "Gradient Descent",
    "Feature Lab",
    "Model Arena",
    "Generalisation & Regularisation",
    "Diagnose",
]
MULTI_FEATURE_DATASETS = [
    "FuelConsumption CO2",
    "Energy Efficiency",
    "Wine Quality (red)",
    "Diabetes",
    "Startup Profit",
    "Student wellbeing",
]
DIAGNOSTIC_MYSTERIES = [
    "Mystery A",
    "Mystery B",
    "Mystery C",
    "Mystery D",
    "Mystery E",
    "Mystery F",
    "Mystery G",
]


def _new_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def _open_workspace(workspace: str) -> AppTest:
    app = _new_app()
    app.radio(key="workspace").set_value(workspace).run(timeout=120)
    return app


def _assert_clean_render(app: AppTest) -> None:
    assert not app.exception
    assert not app.error


def test_initial_render_has_masthead_navigation_and_no_exception():
    app = _new_app()

    _assert_clean_render(app)
    assert app.radio(key="workspace").options == WORKSPACES
    assert app.radio(key="workspace").value == WORKSPACES[0]
    assert len(app.toggle) == 0
    assert "Metrics Under Pressure" not in app.radio(key="workspace").options
    assert any("Regression Playground" in element.value for element in app.markdown)
    assert len(app.get("plotly_chart")) == 1


@pytest.mark.parametrize("workspace", WORKSPACES)
def test_every_workspace_renders_without_exception(workspace):
    app = _open_workspace(workspace)

    _assert_clean_render(app)
    assert app.radio(key="workspace").value == workspace
    assert len(app.get("plotly_chart")) >= 1


def test_gradient_descent_excessive_rate_is_handled_as_a_teaching_result():
    app = _open_workspace("Gradient Descent")

    app.selectbox(key="gd_preset").set_value("Too large · 2.50").run(timeout=120)
    app.slider(key="gd_iterations").set_value(500).run(timeout=120)

    _assert_clean_render(app)
    assert any("steps are too large" in message.value for message in app.warning)
    assert next(
        metric.value for metric in app.metric if metric.label == "Status"
    ) == "Diverging"
    cost_chart = json.loads(app.get("plotly_chart")[1].proto.spec)
    trace_names = [trace.get("name", "") for trace in cost_chart["data"]]
    assert not any("α = 2.50" in name for name in trace_names)
    assert not any("continues upward" in name for name in trace_names)
    assert not any(name.startswith("Selected ·") for name in trace_names)
    assert not any(name.startswith("selected α") for name in trace_names)
    assert any(
        "intentionally omitted" in caption.value
        for caption in app.caption
    )


@pytest.mark.parametrize("pattern", list(ru.GRADIENT_DESCENT_PATTERNS))
def test_gradient_descent_renders_every_data_pattern(pattern):
    app = _open_workspace("Gradient Descent")

    app.selectbox(key="gd_pattern").set_value(pattern).run(timeout=120)

    _assert_clean_render(app)
    assert app.selectbox(key="gd_pattern").value == pattern
    assert len(app.get("plotly_chart")) == 2
    metric_labels = {metric.label for metric in app.metric}
    assert "θ₀ · raw-x intercept" in metric_labels
    assert "θ₁ · raw-x coefficient" in metric_labels


@pytest.mark.parametrize(
    ("preset", "selected_alpha"),
    [
        ("Too small · 0.01", "α = 0.01"),
        ("Useful · 0.30", "α = 0.30"),
    ],
)
def test_gradient_descent_cost_chart_compares_and_highlights_learning_rates(
    preset, selected_alpha
):
    app = _open_workspace("Gradient Descent")
    app.selectbox(key="gd_preset").set_value(preset).run(timeout=120)

    cost_chart = json.loads(app.get("plotly_chart")[1].proto.spec)
    line_traces = [
        trace for trace in cost_chart["data"] if trace.get("mode") == "lines"
    ]
    names = [trace["name"] for trace in line_traces]

    assert len(line_traces) == 2
    assert any("α = 0.01" in name for name in names)
    assert any("α = 0.30" in name for name in names)
    assert not any("α = 2.50" in name for name in names)
    selected = [trace for trace in line_traces if trace["name"].startswith("Selected")]
    assert len(selected) == 1
    assert selected_alpha in selected[0]["name"]
    assert selected[0]["line"]["width"] > max(
        trace["line"]["width"] for trace in line_traces if trace not in selected
    )
    selected_markers = [
        trace
        for trace in cost_chart["data"]
        if trace.get("name", "").startswith("selected α")
    ]
    assert len(selected_markers) == 1
    assert cost_chart["layout"]["yaxis"].get("type", "linear") == "linear"


def test_gradient_descent_adds_a_stable_custom_rate_without_the_large_reference():
    app = _open_workspace("Gradient Descent")

    app.selectbox(key="gd_preset").set_value("Custom").run(timeout=120)
    app.slider(key="gd_alpha").set_value(0.8).run(timeout=120)

    _assert_clean_render(app)
    cost_chart = json.loads(app.get("plotly_chart")[1].proto.spec)
    line_traces = [
        trace for trace in cost_chart["data"] if trace.get("mode") == "lines"
    ]
    names = [trace["name"] for trace in line_traces]
    assert len(line_traces) == 3
    assert any(name.startswith("Selected · α = 0.8") for name in names)
    assert not any("α = 2.50" in name for name in names)


@pytest.mark.parametrize(
    ("workspace", "button_key", "minimum_plots"),
    [
        ("Feature Lab", "feature_final_reveal_button", 3),
        ("Model Arena", "arena_final_reveal_button", 1),
        ("Generalisation & Regularisation", "complexity_reveal_button", 2),
        ("Diagnose", "diagnostic_reveal_button", 5),
    ],
)
def test_staged_reveals_run_without_exception(workspace, button_key, minimum_plots):
    app = _open_workspace(workspace)

    app.button(key=button_key).click().run(timeout=120)

    _assert_clean_render(app)
    assert len(app.get("plotly_chart")) >= minimum_plots


@pytest.mark.parametrize("dataset", MULTI_FEATURE_DATASETS)
def test_feature_lab_renders_each_course_dataset(dataset):
    app = _open_workspace("Feature Lab")

    app.selectbox(key="feature_dataset").set_value(dataset).run(timeout=120)

    _assert_clean_render(app)
    assert app.selectbox(key="feature_dataset").value == dataset
    assert len(app.get("plotly_chart")) >= 2


@pytest.mark.parametrize("mystery", DIAGNOSTIC_MYSTERIES)
def test_each_diagnostic_mystery_renders_all_linked_views(mystery):
    app = _open_workspace("Diagnose")

    app.selectbox(key="diagnostic_mystery").set_value(mystery).run(timeout=120)

    _assert_clean_render(app)
    assert len(app.get("plotly_chart")) == 5
    hidden_text = "\n".join(str(markdown.value) for markdown in app.markdown)
    assert "Why this clue fits:" not in hidden_text

    app.button(key="diagnostic_reveal_button").click().run(timeout=120)

    _assert_clean_render(app)
    revealed_text = "\n".join(str(markdown.value) for markdown in app.markdown)
    assert "Why this clue fits:" in revealed_text
    assert "Strongest figure(s):" in revealed_text
    assert "What to inspect:" in revealed_text


def test_regularisation_investigation_and_final_reveal_run_cleanly():
    app = _open_workspace("Generalisation & Regularisation")

    app.radio(key="generalisation_section").set_value("Ridge & Lasso").run(timeout=120)
    _assert_clean_render(app)
    assert len(app.get("plotly_chart")) == 2

    app.button(key="regularisation_final_reveal_button").click().run(timeout=120)

    _assert_clean_render(app)
    assert any("Locked final test" in message.value for message in app.success)


def test_feature_ranking_reveal_runs_without_serialising_locked_split_metadata():
    app = _open_workspace("Feature Lab")

    app.button(key="feature_ranking_button").click().run(timeout=120)

    _assert_clean_render(app)
    assert len(app.dataframe) == 1
    assert "CV R² mean" in app.dataframe[0].value.columns


def test_feature_lab_cv_metrics_keep_mean_and_spread_fully_visible():
    app = _open_workspace("Feature Lab")

    metrics = {metric.label: metric for metric in app.metric}
    for label in ("CV R²", "CV RMSE"):
        assert "..." not in metrics[label].value
        assert "±" in metrics[label].delta
        assert metrics[label].delta.endswith(" SD")


def test_model_arena_exposes_every_available_model_without_a_second_mode():
    app = _open_workspace("Model Arena")

    assert app.selectbox(key="arena_model_A").options == ru.MODEL_CHOICES
    assert app.selectbox(key="arena_model_B").options == ru.MODEL_CHOICES
    app.selectbox(key="arena_model_A").set_value("Decision Tree").run(timeout=120)

    _assert_clean_render(app)
    assert app.selectbox(key="arena_model_A").value == "Decision Tree"


def test_model_arena_model_b_curve_is_solid_blue():
    app = _open_workspace("Model Arena")

    chart_spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    line_traces = {
        trace["name"]: trace
        for trace in chart_spec["data"]
        if trace.get("mode") == "lines"
    }
    model_b = line_traces[app.selectbox(key="arena_model_B").value]
    model_a = line_traces[app.selectbox(key="arena_model_A").value]

    assert model_b["line"]["dash"] == "solid"
    assert model_b["line"]["color"] == theme.CB_BLUE
    assert model_b["line"]["color"] != model_a["line"]["color"]


def test_fit_attempt_is_logged_immediately():
    app = _new_app()

    app.button(key="fit_log").click().run()

    _assert_clean_render(app)
    assert len(app.session_state["history"]) == 1
    assert app.session_state["history"][0]["workspace"] == "Fit the Line"
    assert any("Attempt added" in message.value for message in app.success)
    assert any("1 logged result" in caption.value for caption in app.sidebar.caption)
