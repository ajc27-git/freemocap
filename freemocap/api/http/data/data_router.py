import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from freemocap.system.default_paths import get_recording_path_in_default_recordings_folder
from freemocap.utilities.download_sample_data import DATASETS, DatasetDownloadResult, download_dataset

logger = logging.getLogger(__name__)

data_router = APIRouter(prefix="/data", tags=["Data"])


class DatasetInfo(BaseModel):
    key: str
    recording_name: str
    menu_label: str
    description: str
    path: str
    available_locally: bool


class DatasetDownloadResponse(BaseModel):
    success: bool
    dataset_key: str
    recording_name: str
    recording_path: str
    recordings_folder_path: str
    downloaded: bool

    @classmethod
    def from_result(cls, result: DatasetDownloadResult) -> "DatasetDownloadResponse":
        return cls(
            success=True,
            dataset_key=result.dataset_key,
            recording_name=result.recording_name,
            recording_path=result.recording_path,
            recordings_folder_path=result.recordings_folder_path,
            downloaded=result.downloaded,
        )


@data_router.get("/datasets")
def list_datasets() -> list[DatasetInfo]:
    """List the datasets FreeMoCap can download, and whether they are already on disk."""
    datasets: list[DatasetInfo] = []
    for dataset in DATASETS.values():
        recording_path = get_recording_path_in_default_recordings_folder(dataset.recording_name)
        datasets.append(
            DatasetInfo(
                key=dataset.key,
                recording_name=dataset.recording_name,
                menu_label=dataset.menu_label,
                description=dataset.description,
                path=recording_path,
                available_locally=Path(recording_path).is_dir(),
            )
        )
    return datasets


@data_router.post("/datasets/{dataset_key}/download")
def download_dataset_endpoint(dataset_key: str, force: bool = False) -> DatasetDownloadResponse:
    """Download `dataset_key` from GitHub and extract it into the recordings folder.

    This is a blocking (`def`) endpoint on purpose: FastAPI runs it in a worker thread, so
    the sizeable download and unzip do not block the event loop. If the recording already
    exists locally nothing is downloaded, unless `force=true`.
    """
    try:
        result = download_dataset(dataset_key, force=force)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to download dataset {dataset_key!r}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return DatasetDownloadResponse.from_result(result)
