import logging
from pathlib import Path
from multiprocessing import Queue
from typing import Optional

from skellytracker.scripts.blendshapes_to_csv import blendshapes_to_csv

from freemocap.data_layer.recording_models.post_processing_parameter_models import ProcessingParameterModel
from freemocap.system.logging.configure_logging import log_view_logging_format_string
from freemocap.system.logging.queue_logger import DirectQueueHandler

logger = logging.getLogger(__name__)


def run_face_blendshapes_pipeline(
        processing_parameters: ProcessingParameterModel,
        queue: Optional[Queue] = None,
):
    if queue:
        handler = DirectQueueHandler(queue)
        handler.setFormatter(logging.Formatter(fmt=log_view_logging_format_string, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)

    recording_folder_path = Path(processing_parameters.recording_info_model.path)
    face_camera_folder = recording_folder_path / "face_camera"
    
    if not face_camera_folder.exists():
        logger.warning(f"Face camera folder not found at {face_camera_folder}. Skipping face blendshapes.")
        return

    # Find the first mp4 file
    mp4_files = list(face_camera_folder.glob("*.mp4"))
    if not mp4_files:
        logger.warning(f"No .mp4 files found in {face_camera_folder}. Skipping face blendshapes.")
        return

    input_facecam_path = mp4_files[0]
    input_file_stem = input_facecam_path.stem

    output_folder = recording_folder_path / "output_data" / "face_blendshapes"
    output_folder.mkdir(parents=True, exist_ok=True)

    annotated_output_path = output_folder / f"{input_file_stem}_annotated.mp4"
    csv_output_path = output_folder / f"{input_file_stem}.csv"

    logger.info(f"Running face blendshapes tracker for {input_file_stem}")
        
    try:
        blendshapes_to_csv(
            input_video_filepath=input_facecam_path,
            output_video_filepath=annotated_output_path,
            output_csv_filepath=csv_output_path,
        )
        logger.info("Face blendshapes tracker completed successfully.")
    except Exception as e:
        logger.error(f"Face blendshapes tracker failed: {e}")
        raise e
