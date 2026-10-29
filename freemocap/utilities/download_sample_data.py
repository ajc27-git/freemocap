"""Download FreeMoCap's public datasets and extract them into the recordings folder.

The archives are hosted as GitHub release assets (``freemocap/skellysamples``). Each zip
contains a single top-level folder that is extracted verbatim into the default recordings
folder, so ``<recordings>/<recording_name>`` becomes a ready-to-load recording.

This replaces the Qt-era helper that imported the since-removed
``freemocap.system.paths_and_filenames`` package.
"""
from __future__ import annotations

import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from freemocap.system.default_paths import (
    FREEMOCAP_SAMPLE_DATA_RECORDING_NAME,
    FREEMOCAP_TEST_DATA_RECORDING_NAME,
    get_default_freemocap_recordings_path,
)

logger = logging.getLogger(__name__)

DATASET_TEST_KEY = "test"
DATASET_SAMPLE_KEY = "sample"

_CHUNK_SIZE_BYTES = 1024 * 1024
# (connect timeout, read timeout) — the sample dataset is ~170 MB
_DOWNLOAD_TIMEOUT = (10, 300)


@dataclass(frozen=True)
class Dataset:
    key: str
    recording_name: str
    url: str
    menu_label: str
    description: str = ""


DATASETS: dict[str, Dataset] = {
    DATASET_TEST_KEY: Dataset(
        key=DATASET_TEST_KEY,
        recording_name=FREEMOCAP_TEST_DATA_RECORDING_NAME,
        url="https://github.com/freemocap/skellysamples/releases/download/test_data_v06_09_25/freemocap_test_data.zip",
        menu_label="Load Test Data",
        description="3 cameras, ~200 frames (37 MB)",
    ),
    DATASET_SAMPLE_KEY: Dataset(
        key=DATASET_SAMPLE_KEY,
        recording_name=FREEMOCAP_SAMPLE_DATA_RECORDING_NAME,
        url="https://github.com/freemocap/skellysamples/releases/download/sample_data_v06_12_25/freemocap_sample_data.zip",
        menu_label="Load Sample Data",
        description="3 cameras, ~1000 frames (167 MB)",
    ),
}


@dataclass(frozen=True)
class DatasetDownloadResult:
    dataset_key: str
    recording_name: str
    recording_path: str
    recordings_folder_path: str
    downloaded: bool


def get_dataset(dataset_key: str) -> Dataset:
    """Look up a dataset by key, raising a helpful ``ValueError`` if it is unknown."""
    dataset = DATASETS.get(dataset_key)
    if dataset is None:
        raise ValueError(f"Unknown dataset {dataset_key!r}. Options: {sorted(DATASETS)}")
    return dataset


def download_dataset(
    dataset_key: str,
    recordings_folder_path: str | Path | None = None,
    force: bool = False,
) -> DatasetDownloadResult:
    """Download `dataset_key`, extract it into the recordings folder, and return the result.

    If the recording folder already exists and `force` is False the download is skipped, so
    this also doubles as "make sure this dataset is available locally".
    """
    dataset = get_dataset(dataset_key)
    destination = Path(recordings_folder_path or get_default_freemocap_recordings_path()).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    recording_path = destination / dataset.recording_name
    already_present = recording_path.is_dir()

    if already_present and not force:
        logger.info(f"'{dataset.recording_name}' already exists at {recording_path} - skipping download")
    else:
        _download_and_extract(zip_file_url=dataset.url, destination=destination)
        if not recording_path.is_dir():
            raise RuntimeError(
                f"Downloaded {dataset.url!r}, but the expected folder {recording_path} is missing. "
                f"Contents of {destination}: {sorted(path.name for path in destination.iterdir())}"
            )

    return DatasetDownloadResult(
        dataset_key=dataset.key,
        recording_name=dataset.recording_name,
        recording_path=str(recording_path),
        recordings_folder_path=str(destination),
        downloaded=(not already_present) or force,
    )


def download_test_data(**kwargs) -> DatasetDownloadResult:
    """Download/extract the small ``freemocap_test_data`` recording (~200 frames)."""
    return download_dataset(DATASET_TEST_KEY, **kwargs)


def download_sample_data(**kwargs) -> DatasetDownloadResult:
    """Download/extract the larger ``freemocap_sample_data`` recording (~1000 frames)."""
    return download_dataset(DATASET_SAMPLE_KEY, **kwargs)


def get_dataset_recording_path(dataset_key: str, download_if_needed: bool = True) -> str:
    """Path to a dataset recording, downloading it first if it is not present."""
    dataset = get_dataset(dataset_key)
    recording_path = Path(get_default_freemocap_recordings_path()) / dataset.recording_name
    if not recording_path.is_dir():
        if not download_if_needed:
            raise FileNotFoundError(
                f"'{dataset.recording_name}' not found at {recording_path} (and `download_if_needed` is False)"
            )
        download_dataset(dataset_key)
    return str(recording_path)


def _download_and_extract(zip_file_url: str, destination: Path) -> None:
    logger.info(f"Downloading {zip_file_url} ...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary_file:
        temporary_zip_path = Path(temporary_file.name)

    try:
        with requests.get(zip_file_url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length") or 0)
            next_log_fraction = 0.1
            downloaded_bytes = 0
            with open(temporary_zip_path, "wb") as zip_file:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE_BYTES):
                    if not chunk:
                        continue
                    zip_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    if total_bytes > 0 and downloaded_bytes / total_bytes >= next_log_fraction:
                        logger.info(
                            f"  ... {downloaded_bytes / total_bytes:.0%} "
                            f"of {total_bytes / 1e6:.0f} MB downloaded"
                        )
                        next_log_fraction += 0.1

        logger.info(f"Download complete ({downloaded_bytes / 1e6:.1f} MB) - extracting to {destination}")
        _safe_extract_zip(temporary_zip_path, destination)
        logger.info(f"Extracted to {destination}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {zip_file_url!r}: {e}")
        raise
    finally:
        temporary_zip_path.unlink(missing_ok=True)


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract `zip_path` into `destination`, rejecting members that escape it (zip-slip)."""
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination):
                raise RuntimeError(f"Refusing to extract {member.filename!r}: outside of {destination}")
        archive.extractall(destination)


if __name__ == "__main__":
    import sys

    dataset_key = sys.argv[1] if len(sys.argv) > 1 else DATASET_TEST_KEY
    download_result = download_dataset(dataset_key)
    print(f"Dataset {download_result.dataset_key!r} available at {download_result.recording_path}")
