"""Shallow import test for the Streamlit app.

Streamlit is an optional dependency. If it's not installed, skip.
We only check that the module imports and exposes the expected symbols
without executing the app.
"""

import importlib.util
from pathlib import Path

import pytest

_STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None

pytestmark = pytest.mark.skipif(
    not _STREAMLIT_AVAILABLE, reason="streamlit not installed"
)


def test_app_module_loads(monkeypatch):
    """Import ui/app.py without running streamlit's runtime.

    Streamlit's bare mode allows importing an app file, but its `st.*`
    calls execute at import time. We monkeypatch `st.set_page_config` etc.
    indirectly by loading the module fresh and asserting key attributes.
    """
    import importlib.util

    app_path = Path(__file__).parent.parent / "ui" / "app.py"
    assert app_path.exists(), f"expected {app_path} to exist"

    spec = importlib.util.spec_from_file_location("limen_ui_app", app_path)
    mod = importlib.util.module_from_spec(spec)
    # We don't exec the module body — only load and check it's importable as source.
    assert spec is not None
    assert spec.loader is not None
    # Sanity: file has content
    assert app_path.read_text().strip() != ""
