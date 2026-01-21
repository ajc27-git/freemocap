import logging
import multiprocessing
from pathlib import Path

import numpy as np
from skellytracker.trackers.base_tracker.model_info import ModelInfo
from skellytracker.process_folder_of_videos import process_folder_of_videos
from skellytracker.trackers.color_tracker.color_tracker import ColorTracker
from skellytracker.trackers.color_tracker.color_tracker_model_info import ColorTrackerModelInfo

from freemocap.data_layer.recording_models.post_processing_parameter_models import (
    ProcessingParameterModel,
    ColorMarkerConfig,
)
from freemocap.system.logging.configure_logging import log_view_logging_format_string
from freemocap.system.logging.queue_logger import DirectQueueHandler
from freemocap.system.paths_and_filenames.file_and_folder_names import (
    LOG_VIEW_PROGRESS_BAR_STRING,
    RAW_DATA_FOLDER_NAME,
    COLOR_MARKERS_2D_NPY_FILE_NAME,
)

logger = logging.getLogger(__name__)


# Update the run_color_tracking_pipeline function to create annotated videos
def run_color_tracking_pipeline(
    processing_parameters: ProcessingParameterModel,
    kill_event: multiprocessing.Event,
    queue: multiprocessing.Queue,
    use_tqdm: bool,
) -> np.ndarray:
    """
    Run color tracking pipeline to detect colored markers in videos.
    
    Returns:
        color_markers_2d_data: Array of shape [num_cameras, num_frames, num_color_markers, 3]
                              where 3 is (x, y, confidence)
    """
    if not processing_parameters.color_tracker_parameters_model.run_color_tracking:
        logger.info("Color tracking is disabled, skipping...")
        return None
    
    if queue:
        handler = DirectQueueHandler(queue)
        handler.setFormatter(logging.Formatter(fmt=log_view_logging_format_string, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
    
    # Check if we have any enabled markers
    enabled_markers = [
        config for config in processing_parameters.color_tracker_parameters_model.marker_configs 
        if config.enabled
    ]
    
    if not enabled_markers:
        logger.info("No color markers enabled, skipping color tracking...")
        return None
    
    logger.info(f"Starting color tracking for {len(enabled_markers)} markers...")
    logger.info(LOG_VIEW_PROGRESS_BAR_STRING)
    
    # Create color tracker model info instance
    color_tracker_model_info = ColorTrackerModelInfo(
        marker_configs=processing_parameters.color_tracker_parameters_model.marker_configs,
        use_morphological_ops=processing_parameters.color_tracker_parameters_model.use_morphological_ops,
    )
    
    # Create annotated videos folder
    annotated_videos_folder = Path(processing_parameters.recording_info_model.output_data_folder_path) / "annotated_videos_color_tracking"
    annotated_videos_folder.mkdir(exist_ok=True, parents=True)
    
    # Run color tracking
    color_markers_2d_data = run_color_tracking(
        model_info=color_tracker_model_info,
        synchronized_videos_folder_path=Path(
            processing_parameters.recording_info_model.synchronized_videos_folder_path
        ),
        output_data_folder_path=Path(processing_parameters.recording_info_model.output_data_folder_path)
        / RAW_DATA_FOLDER_NAME,
        annotated_videos_folder_path=annotated_videos_folder,
        num_processes=processing_parameters.color_tracker_parameters_model.num_processes,
    )
    
    # Save the color markers 2D data
    output_path = Path(processing_parameters.recording_info_model.output_data_folder_path) / RAW_DATA_FOLDER_NAME / COLOR_MARKERS_2D_NPY_FILE_NAME
    np.save(output_path, color_markers_2d_data)
    logger.info(f"Saved color markers 2D data to: {output_path}")
    
    return color_markers_2d_data


def run_color_tracking(
    model_info: ModelInfo,
    synchronized_videos_folder_path: Path,
    output_data_folder_path: Path,
    annotated_videos_folder_path: Path = None,
    num_processes: int = 1,
) -> np.ndarray:
    """
    Run color tracking on a folder of synchronized videos.
    """
    # Get marker_configs from model_info
    marker_configs = model_info.marker_configs
    use_morphological_ops = model_info.use_morphological_ops
    
    # Create color tracker instance
    color_tracker = ColorTracker(
        marker_configs=marker_configs,
        use_morphological_ops=use_morphological_ops,
    )
    
    # Process videos using skellytracker
    color_markers_2d_data = process_folder_of_videos(
        model_info=model_info,
        tracking_params=color_tracker,
        synchronized_video_path=synchronized_videos_folder_path,
        output_folder_path=output_data_folder_path,
        annotated_video_path=annotated_videos_folder_path,
        num_processes=num_processes,
    )
    
    return color_markers_2d_data
