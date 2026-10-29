import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import freemocap.api.http.data.data_router as data_router_module
from freemocap.api.http.data.data_router import data_router
from freemocap.utilities.download_sample_data import (
    DATASET_SAMPLE_KEY,
    DATASET_TEST_KEY,
    DatasetDownloadResult,
)


@pytest.fixture
def client():
    test_app = FastAPI()
    test_app.include_router(data_router)
    return TestClient(test_app)


@pytest.fixture
def recordings_folder(tmp_path, monkeypatch):
    """Point the recordings folder at `tmp_path` so nothing touches the real data folder."""
    monkeypatch.setattr(
        data_router_module,
        "get_recording_path_in_default_recordings_folder",
        lambda recording_name: str(tmp_path / recording_name),
    )
    return tmp_path


def test_download_extracts_into_recordings_folder(client, recordings_folder, monkeypatch):
    captured: dict = {}

    def fake_download_dataset(dataset_key: str, recordings_folder_path=None, force: bool = False):
        captured["dataset_key"] = dataset_key
        captured["force"] = force
        recording_path = recordings_folder / "freemocap_test_data"
        recording_path.mkdir()
        return DatasetDownloadResult(
            dataset_key=dataset_key,
            recording_name=recording_path.name,
            recording_path=str(recording_path),
            recordings_folder_path=str(recordings_folder),
            downloaded=True,
        )

    monkeypatch.setattr(data_router_module, "download_dataset", fake_download_dataset)

    response = client.post(f"/data/datasets/{DATASET_TEST_KEY}/download")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "success": True,
        "dataset_key": DATASET_TEST_KEY,
        "recording_name": "freemocap_test_data",
        "recording_path": str(recordings_folder / "freemocap_test_data"),
        "recordings_folder_path": str(recordings_folder),
        "downloaded": True,
    }
    assert captured == {"dataset_key": DATASET_TEST_KEY, "force": False}


def test_download_forwards_force_flag(client, recordings_folder, monkeypatch):
    captured: dict = {}

    def fake_download_dataset(dataset_key: str, recordings_folder_path=None, force: bool = False):
        captured["force"] = force
        return DatasetDownloadResult(
            dataset_key=dataset_key,
            recording_name="freemocap_sample_data",
            recording_path=str(recordings_folder / "freemocap_sample_data"),
            recordings_folder_path=str(recordings_folder),
            downloaded=force,
        )

    monkeypatch.setattr(data_router_module, "download_dataset", fake_download_dataset)

    response = client.post(f"/data/datasets/{DATASET_SAMPLE_KEY}/download?force=true")

    assert response.status_code == 200
    assert captured == {"force": True}


def test_unknown_dataset_returns_404(client, recordings_folder):
    response = client.post("/data/datasets/not-a-dataset/download")

    assert response.status_code == 404
    assert "Unknown dataset" in response.json()["detail"]


def test_list_datasets_reports_local_availability(client, recordings_folder):
    (recordings_folder / "freemocap_test_data").mkdir()

    response = client.get("/data/datasets")

    assert response.status_code == 200
    by_key = {entry["key"]: entry for entry in response.json()}
    assert set(by_key) == {DATASET_TEST_KEY, DATASET_SAMPLE_KEY}
    assert by_key[DATASET_TEST_KEY]["available_locally"] is True
    assert by_key[DATASET_SAMPLE_KEY]["available_locally"] is False
    assert by_key[DATASET_SAMPLE_KEY]["path"] == str(recordings_folder / "freemocap_sample_data")
