from types import SimpleNamespace

from src.launcher.ui import LauncherUI, _PyArrowImportBlocker


def test_launcher_ui_imports_gradio_with_pyarrow_blocker_removed():
    blocker_names = {type(finder).__name__ for finder in __import__("sys").meta_path}

    assert _PyArrowImportBlocker.__name__ not in blocker_names


def test_launcher_ui_builds_interface_without_model_dependencies():
    search_engine = SimpleNamespace(indexer=SimpleNamespace(file_metadata=[]))
    ui = LauncherUI(search_engine)

    interface = ui.create_interface()

    assert interface.__class__.__name__ == "Blocks"
