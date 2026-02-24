import logging
import multiprocessing
from pathlib import Path
from typing import Optional
import json
import numpy as np

from skellytracker.trackers.base_tracker.model_info import ModelInfo
from skellytracker.process_folder_of_videos import process_folder_of_videos
from skellytracker.trackers.yolo_object_tracker.yolo_object_tracker import YOLOObjectTracker
from skellytracker.trackers.yolo_object_tracker.yolo_object_model_info import YOLOObjectModelInfo

from freemocap.data_layer.recording_models.post_processing_parameter_models import (
    ProcessingParameterModel,
)
from freemocap.system.logging.configure_logging import log_view_logging_format_string
from freemocap.system.logging.queue_logger import DirectQueueHandler
from freemocap.system.paths_and_filenames.file_and_folder_names import (
    LOG_VIEW_PROGRESS_BAR_STRING,
    YOLO_OBJECT_TRACKING_FOLDER_NAME,
    YOLO_OBJECT_MARKERS_2D_NPY_FILE_NAME,
)

logger = logging.getLogger(__name__)


def run_yolo_object_tracking_pipeline(
    processing_parameters: ProcessingParameterModel,
    kill_event: multiprocessing.Event,
    queue: multiprocessing.Queue,
    use_tqdm: bool,
) -> Optional[np.ndarray]:
    """
    Run YOLO object tracking pipeline to detect objects in videos.
    """
    if not processing_parameters.yolo_object_tracker_parameters_model.run_yolo_object_tracker:
        logger.info("YOLO object tracking is disabled, skipping...")
        return None
    
    if queue:
        handler = DirectQueueHandler(queue)
        handler.setFormatter(logging.Formatter(fmt=log_view_logging_format_string, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
    
    logger.info("Starting YOLO object tracking...")
    logger.info(LOG_VIEW_PROGRESS_BAR_STRING)
    
    # Create YOLO object tracker model info instance
    yolo_object_model_info = YOLOObjectModelInfo(
        model_path=processing_parameters.yolo_object_tracker_parameters_model.custom_model_path,
        model_size=processing_parameters.yolo_object_tracker_parameters_model.model_size,
        # person_only=processing_parameters.yolo_object_tracker_parameters_model.person_only,
        confidence_threshold=processing_parameters.yolo_object_tracker_parameters_model.confidence_threshold,
        # classes_to_track=processing_parameters.yolo_object_tracker_parameters_model.classes_to_track,
        # max_detections=processing_parameters.yolo_object_tracker_parameters_model.max_detections,
    )
    
    # Save YOLO object model info to json
    yolo_object_model_info_path = (
        Path(processing_parameters.recording_info_model.output_data_folder_path)
        / YOLO_OBJECT_TRACKING_FOLDER_NAME
        / "yolo_object_model_info.json"
    )
    yolo_object_model_info_path.parent.mkdir(exist_ok=True, parents=True)
    with open(yolo_object_model_info_path, "w") as f:
        json.dump(yolo_object_model_info.get_config(), f, indent=4)
    logger.info(f"Saved YOLO object model info to: {yolo_object_model_info_path}")

    # Create annotated videos folder
    annotated_videos_folder = Path(processing_parameters.recording_info_model.output_data_folder_path) / YOLO_OBJECT_TRACKING_FOLDER_NAME / "annotated_videos_yolo_object_tracking"
    annotated_videos_folder.mkdir(exist_ok=True, parents=True)
    
    # Run YOLO object tracking
    yolo_objects_2d_data = run_yolo_object_tracking(
        model_info=yolo_object_model_info,
        synchronized_videos_folder_path=Path(
            processing_parameters.recording_info_model.synchronized_videos_folder_path
        ),
        output_data_folder_path=Path(processing_parameters.recording_info_model.output_data_folder_path)
        / YOLO_OBJECT_TRACKING_FOLDER_NAME,
        annotated_videos_folder_path=annotated_videos_folder,
        # num_processes=processing_parameters.yolo_object_tracker_parameters_model.num_processes,
    )
    
    # Save the YOLO objects 2D data
    output_path = Path(processing_parameters.recording_info_model.output_data_folder_path) / YOLO_OBJECT_TRACKING_FOLDER_NAME / YOLO_OBJECT_MARKERS_2D_NPY_FILE_NAME
    np.save(output_path, yolo_objects_2d_data)
    logger.info(f"Saved YOLO objects 2D data to: {output_path}")
    
    return yolo_objects_2d_data


def run_yolo_object_tracking(
    model_info: ModelInfo,
    synchronized_videos_folder_path: Path,
    output_data_folder_path: Path,
    annotated_videos_folder_path: Optional[Path] = None,
    num_processes: int = 1,
) -> np.ndarray:
    """
    Run YOLO object tracking on a folder of synchronized videos.
    """
    # Extract parameters from model_info
    model_path = model_info.model_path
    model_size = model_info.model_size
    # person_only = model_info.person_only
    confidence_threshold = model_info.confidence_threshold
    # classes_to_track = model_info.classes_to_track
    # max_detections = model_info.max_detections
    
    # Create YOLO object tracker instance
    yolo_object_tracker = YOLOObjectTracker(
        model_path=model_path,
        model_size=model_size,
        # person_only=person_only,
        confidence_threshold=confidence_threshold,
        # classes_to_track=classes_to_track,
        # max_detections=max_detections,
    )
    
    # Process videos using skellytracker
    yolo_objects_2d_data = process_folder_of_videos(
        model_info=model_info,
        tracking_params=yolo_object_tracker,
        synchronized_video_path=synchronized_videos_folder_path,
        output_folder_path=output_data_folder_path,
        annotated_video_path=annotated_videos_folder_path,
        num_processes=num_processes,
    )
    
    return yolo_objects_2d_data


# Example usage
if __name__ == "__main__":
    # Example configuration
    from freemocap.data_layer.recording_models.post_processing_parameter_models import (
        YOLOObjectTrackerParametersModel,
    )
    
    # Create a mock processing parameters object
    class MockProcessingParameters:
        def __init__(self):
            self.yolo_object_tracker_parameters_model = YOLOObjectTrackerParametersModel(
                run_yolo_object_tracking=True,
                model_path=None,
                model_size="medium",
                person_only=False,
                confidence_threshold=0.5,
                classes_to_track=None,
                max_detections=1,
                num_processes=4,
            )
            self.recording_info_model = type('obj', (object,), {
                'output_data_folder_path': '/path/to/output',
                'synchronized_videos_folder_path': '/path/to/synchronized_videos'
            })()
    
    # Create mock multiprocessing objects
    import multiprocessing
    kill_event = multiprocessing.Event()
    queue = multiprocessing.Queue()
    
    # Run the pipeline (this would normally be called from FreeMoCap)
    yolo_objects_2d_data = run_yolo_object_tracking_pipeline(
        processing_parameters=MockProcessingParameters(),
        kill_event=kill_event,
        queue=queue,
        use_tqdm=True,
    )