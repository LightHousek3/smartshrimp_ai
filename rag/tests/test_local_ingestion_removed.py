import importlib.util


def test_local_ingestion_module_is_removed():
    assert importlib.util.find_spec("app.ingestion.ingest") is None
