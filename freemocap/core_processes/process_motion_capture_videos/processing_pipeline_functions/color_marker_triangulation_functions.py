import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np

from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration.get_anipose_calibration_object import (
    load_anipose_calibration_toml_from_path,
)
from freemocap.core_processes.capture_volume_calibration.triangulate_3d_data import triangulate_3d_data
from freemocap.core_processes.capture_volume_calibration.save_3d_data_to_npy import (
    save_3d_data_to_npy,
)
from freemocap.data_layer.recording_models.post_processing_parameter_models import ProcessingParameterModel
from freemocap.system.logging.configure_logging import log_view_logging_format_string
from freemocap.system.logging.queue_logger import DirectQueueHandler
from freemocap.system.paths_and_filenames.file_and_folder_names import (
    LOG_VIEW_PROGRESS_BAR_STRING,
    COLOR_MARKERS_2D_NPY_FILE_NAME,
    COLOR_MARKERS_3D_NPY_FILE_NAME,
)
from freemocap.core_processes.post_process_skeleton_data.process_single_camera_skeleton_data import (
    process_single_camera_skeleton_data,
)

logger = logging.getLogger(__name__)


def get_color_markers_triangulated_data(
        color_markers_2d_data: np.ndarray,
        processing_parameters: ProcessingParameterModel,
        kill_event: Optional[multiprocessing.Event] = None,
        queue: Optional[multiprocessing.Queue] = None,
) -> np.ndarray:
    """
    Triangulate 2D color marker data to 3D.
    
    Args:
        color_markers_2d_data: Array of shape [num_cameras, num_frames, num_color_markers, 3]
        processing_parameters: Processing parameters
        kill_event: Event to kill the process
        queue: Logging queue
        
    Returns:
        color_markers_3d_data: Array of shape [num_frames, num_color_markers, 3]
    """
    if color_markers_2d_data is None:
        logger.info("No color markers 2D data to triangulate")
        return None
    
    if queue:
        handler = DirectQueueHandler(queue)
        handler.setFormatter(logging.Formatter(fmt=log_view_logging_format_string, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
    
    # Check if we should skip triangulation
    if not processing_parameters.anipose_triangulate_3d_parameters_model.run_3d_triangulation:
        logger.info("3D triangulation is disabled, skipping color markers triangulation...")
        return None
    
    # Handle single camera case
    if color_markers_2d_data.shape[0] == 1:
        logger.info("Processing single camera color markers data...")
        logger.info(LOG_VIEW_PROGRESS_BAR_STRING)
        
        # Use the existing single camera processing function
        (color_markers_3d_data, _) = process_single_camera_skeleton_data(
            input_image_data_frame_marker_xyz=color_markers_2d_data[0],
            raw_data_folder_path=Path(processing_parameters.recording_info_model.raw_data_folder_path),
            file_prefix="color_markers",  # Use different prefix for color markers
            project_to_z_plane=processing_parameters.anipose_triangulate_3d_parameters_model.flatten_single_camera_data,
        )
        return color_markers_3d_data
    
    logger.info("Triangulating color markers to 3D...")
    logger.info(LOG_VIEW_PROGRESS_BAR_STRING)
    
    # Check for calibration file
    if not processing_parameters.recording_info_model.calibration_toml_check:
        logger.warning(
            f"No calibration file found at: {processing_parameters.recording_info_model.calibration_toml_path}"
        )
        logger.warning("Cannot triangulate color markers without calibration. Skipping...")
        return None
    
    try:
        # Load calibration
        anipose_calibration_object = load_anipose_calibration_toml_from_path(
            camera_calibration_data_toml_path=processing_parameters.recording_info_model.calibration_toml_path,
            save_copy_of_calibration_to_this_path=processing_parameters.recording_info_model.path,
        )
        
        # Extract only x,y coordinates (skip confidence)
        image_2d_data = color_markers_2d_data[:, :, :, :2]
        
        # Triangulate
        (
            color_markers_3d_data,
            reprojection_error_fr_mar,
            reprojection_error_cam_fr_mar,
        ) = triangulate_3d_data(
            anipose_calibration_object=anipose_calibration_object,
            image_2d_data=image_2d_data,
            use_triangulate_ransac=processing_parameters.anipose_triangulate_3d_parameters_model.use_triangulate_ransac_method,
            kill_event=kill_event,
        )
        
        # Save 3D data
        save_color_markers_3d_data(
            color_markers_3d_data=color_markers_3d_data,
            reprojection_error_fr_mar=reprojection_error_fr_mar,
            reprojection_error_cam_fr_mar=reprojection_error_cam_fr_mar,
            raw_data_folder_path=Path(processing_parameters.recording_info_model.raw_data_folder_path),
        )
        
        logger.info(f"Successfully triangulated {color_markers_3d_data.shape[1]} color markers")
        
        return color_markers_3d_data
        
    except Exception as e:
        logger.error(f"Failed to triangulate color markers: {e}")
        if queue:
            queue.put(e)
        return None


def process_single_camera_color_markers_data(
        input_image_data_frame_marker_xyz: np.ndarray,
        raw_data_folder_path: Path,
        project_to_z_plane: bool = True,
) -> np.ndarray:
    """
    Process color markers data from a single camera.
    For single camera, we can't triangulate, so we use 2D data with Z=0.
    """
    num_frames = input_image_data_frame_marker_xyz.shape[0]
    num_markers = input_image_data_frame_marker_xyz.shape[1]
    
    # Create 3D array with Z=0
    color_markers_3d_data = np.zeros((num_frames, num_markers, 3))
    color_markers_3d_data[:, :, :2] = input_image_data_frame_marker_xyz[:, :, :2]  # Copy x,y
    
    # If projecting to Z plane, set all Z to 0 (already done)
    # Otherwise, we could set Z to some constant or leave as is
    
    # Save the data
    save_color_markers_3d_data(
        color_markers_3d_data=color_markers_3d_data,
        reprojection_error_fr_mar=np.zeros((num_frames, num_markers)),
        reprojection_error_cam_fr_mar=np.zeros((1, num_frames, num_markers)),
        raw_data_folder_path=raw_data_folder_path,
    )
    
    return color_markers_3d_data


def save_color_markers_3d_data(
        color_markers_3d_data: np.ndarray,
        reprojection_error_fr_mar: np.ndarray,
        reprojection_error_cam_fr_mar: np.ndarray,
        raw_data_folder_path: Path,
):
    """
    Save color markers 3D data to files.
    """
    # Save main 3D data
    output_path = raw_data_folder_path / COLOR_MARKERS_3D_NPY_FILE_NAME
    np.save(output_path, color_markers_3d_data)
    logger.info(f"Saved color markers 3D data to: {output_path}")
    
    # Save reprojection error data
    reprojection_error_path = raw_data_folder_path / "color_markers_reprojection_error.npy"
    np.save(reprojection_error_path, reprojection_error_fr_mar)
    
    reprojection_error_cam_path = raw_data_folder_path / "color_markers_reprojection_error_cam.npy"
    np.save(reprojection_error_cam_path, reprojection_error_cam_fr_mar)