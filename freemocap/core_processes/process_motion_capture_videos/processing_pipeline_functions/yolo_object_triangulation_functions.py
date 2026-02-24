import logging
import multiprocessing
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration.get_anipose_calibration_object import (
    load_anipose_calibration_toml_from_path,
)
from freemocap.core_processes.capture_volume_calibration.triangulate_3d_data import triangulate_3d_data

from freemocap.data_layer.recording_models.post_processing_parameter_models import ProcessingParameterModel
from freemocap.system.logging.configure_logging import log_view_logging_format_string
from freemocap.system.logging.queue_logger import DirectQueueHandler
from freemocap.system.paths_and_filenames.file_and_folder_names import (
    LOG_VIEW_PROGRESS_BAR_STRING,
    YOLO_OBJECT_TRACKING_FOLDER_NAME,
    YOLO_OBJECT_MARKERS_3D_NPY_FILE_NAME,
)

logger = logging.getLogger(__name__)


def get_yolo_objects_triangulated_data(
    yolo_objects_2d_data: np.ndarray,
    processing_parameters: ProcessingParameterModel,
    kill_event: Optional[multiprocessing.Event] = None,
    queue: Optional[multiprocessing.Queue] = None,
) -> np.ndarray:
    """
    Triangulate 2D YOLO object data to 3D.
    
    Args:
        yolo_objects_2d_data: Array of shape [num_cameras, num_frames, num_objects, 6]
            where 6 is (x, y, confidence, class_id, box_x1, box_y1)
        processing_parameters: Processing parameters
        kill_event: Event to kill the process
        queue: Logging queue
        
    Returns:
        yolo_objects_3d_data: Array of shape [num_frames, num_objects, 6]
            where 6 is (x, y, z, confidence, class_id, reprojection_error)
    """
    if yolo_objects_2d_data is None:
        logger.info("No YOLO objects 2D data to triangulate")
        return None
    
    if queue:
        handler = DirectQueueHandler(queue)
        handler.setFormatter(logging.Formatter(fmt=log_view_logging_format_string, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
    
    # Check if we should skip triangulation
    if not processing_parameters.anipose_triangulate_3d_parameters_model.run_3d_triangulation:
        logger.info("3D triangulation is disabled, skipping YOLO objects triangulation...")
        return None
    
    # Handle single camera case
    if yolo_objects_2d_data.shape[0] == 1:
        logger.info("Processing single camera YOLO objects data...")
        logger.info(LOG_VIEW_PROGRESS_BAR_STRING)
        
        # Use the existing single camera processing function
        (yolo_objects_3d_data, _) = process_single_camera_yolo_objects_data(
            input_image_data_frame_marker_xyz=yolo_objects_2d_data[0],
            data_folder_path=Path(processing_parameters.recording_info_model.output_data_folder_path) / YOLO_OBJECT_TRACKING_FOLDER_NAME,
            project_to_z_plane=processing_parameters.anipose_triangulate_3d_parameters_model.flatten_single_camera_data,
        )
        return yolo_objects_3d_data
    
    logger.info("Triangulating YOLO objects to 3D...")
    logger.info(LOG_VIEW_PROGRESS_BAR_STRING)
    
    # Check for calibration file
    if not processing_parameters.recording_info_model.calibration_toml_check:
        logger.warning(
            f"No calibration file found at: {processing_parameters.recording_info_model.calibration_toml_path}"
        )
        logger.warning("Cannot triangulate YOLO objects without calibration. Skipping...")
        return None
    
    try:
        # Load calibration
        anipose_calibration_object = load_anipose_calibration_toml_from_path(
            camera_calibration_data_toml_path=processing_parameters.recording_info_model.calibration_toml_path,
            save_copy_of_calibration_to_this_path=processing_parameters.recording_info_model.path,
        )
        
        # Extract only x,y coordinates (skip confidence, class_id, box coordinates)
        image_2d_data = yolo_objects_2d_data[:, :, :, :2]
        
        # Triangulate
        (
            yolo_objects_3d_data,
            reprojection_error_fr_mar,
            reprojection_error_cam_fr_mar,
        ) = triangulate_3d_data(
            anipose_calibration_object=anipose_calibration_object,
            image_2d_data=image_2d_data,
            use_triangulate_ransac=processing_parameters.anipose_triangulate_3d_parameters_model.use_triangulate_ransac_method,
            kill_event=kill_event,
        )
        
        # Fill missing frames
        if processing_parameters.yolo_object_tracker_parameters_model.fill_gaps:
            logger.info("Filling missing frames in 3D data...")
            yolo_objects_3d_data = fill_yolo_objects_3d_gaps(yolo_objects_3d_data)
        
        # Save 3D data
        save_yolo_objects_3d_data(
            yolo_objects_3d_data=yolo_objects_3d_data,
            reprojection_error_fr_mar=reprojection_error_fr_mar,
            reprojection_error_cam_fr_mar=reprojection_error_cam_fr_mar,
            data_folder_path=Path(processing_parameters.recording_info_model.output_data_folder_path) / YOLO_OBJECT_TRACKING_FOLDER_NAME,
        )
        
        logger.info(f"Successfully triangulated {yolo_objects_3d_data.shape[1]} YOLO objects")
        return yolo_objects_3d_data
    
    except Exception as e:
        logger.error(f"Failed to triangulate YOLO objects: {e}")
        if queue:
            queue.put(e)
        return None


def process_single_camera_yolo_objects_data(
    input_image_data_frame_marker_xyz: np.ndarray,
    data_folder_path: Path,
    project_to_z_plane: bool = True,
) -> np.ndarray:
    """
    Process YOLO objects data from a single camera.
    For single camera, we can't triangulate, so we use 2D data with Z=0.
    
    Args:
        input_image_data_frame_marker_xyz: Array of shape [num_frames, num_objects, 6]
        raw_data_folder_path: Path to raw data folder
        project_to_z_plane: Whether to project to Z=0 plane
        
    Returns:
        Tuple of (yolo_objects_3d_data, reprojection_error_fr_mar)
    """
    num_frames = input_image_data_frame_marker_xyz.shape[0]
    num_objects = input_image_data_frame_marker_xyz.shape[1]
    
    # Create 3D array
    yolo_objects_3d_data = np.full((num_frames, num_objects, 6), np.nan, dtype=np.float32)
    
    # Copy x,y coordinates and metadata
    yolo_objects_3d_data[:, :, 0] = input_image_data_frame_marker_xyz[:, :, 0]  # x
    yolo_objects_3d_data[:, :, 1] = input_image_data_frame_marker_xyz[:, :, 1]  # y
    
    # Set Z coordinate
    if project_to_z_plane:
        yolo_objects_3d_data[:, :, 2] = 0.0  # Project to Z=0 plane
    else:
        yolo_objects_3d_data[:, :, 2] = np.nan  # Leave Z as NaN
    
    # Copy metadata
    yolo_objects_3d_data[:, :, 3] = input_image_data_frame_marker_xyz[:, :, 2]  # confidence
    yolo_objects_3d_data[:, :, 4] = input_image_data_frame_marker_xyz[:, :, 3]  # class_id
    
    # Set reprojection error to 0 for single camera
    yolo_objects_3d_data[:, :, 5] = 0.0
    
    # Create reprojection error arrays
    reprojection_error_fr_mar = np.zeros((num_frames, num_objects))
    reprojection_error_cam_fr_mar = np.zeros((1, num_frames, num_objects))
    
    # Save the data
    save_yolo_objects_3d_data(
        yolo_objects_3d_data=yolo_objects_3d_data,
        reprojection_error_fr_mar=reprojection_error_fr_mar,
        reprojection_error_cam_fr_mar=reprojection_error_cam_fr_mar,
        data_folder_path=data_folder_path,
    )
    
    return yolo_objects_3d_data, reprojection_error_fr_mar


def fill_yolo_objects_3d_gaps(yolo_objects_3d_data: np.ndarray) -> np.ndarray:
    """
    Fill gaps (NaNs) in 3D YOLO tracking data.
    Uses 'pchip' interpolation (shape-preserving cubic spline) to maintain movement physics.
    Falls back to linear if pchip fails (e.g. not enough data points).
    
    Args:
        yolo_objects_3d_data: Array of shape [num_frames, num_objects, 6]
            (x, y, z, confidence, class_id, reprojection_error)
            
    Returns:
        Gap-filled array of the same shape.
    """
    out_data = yolo_objects_3d_data.copy()
    num_objects = out_data.shape[1]
    
    for obj_idx in range(num_objects):
        coords = out_data[:, obj_idx, :3]
        df = pd.DataFrame(coords)
        
        # Skip if completely empty or no NaNs to fill
        if df.isna().all().all() or not df.isna().any().any():
            continue
            
        try:
            # pchip is a shape-preserving cubic spline, good for physics/movement
            df_filled = df.interpolate(method='pchip', limit_direction='both')
        except Exception as e:
            logger.warning(f"Failed to use 'pchip' interpolation for object {obj_idx}, falling back to 'linear': {e}")
            df_filled = df.interpolate(method='linear', limit_direction='both')
            
        out_data[:, obj_idx, :3] = df_filled.values
        
    return out_data


def save_yolo_objects_3d_data(
    yolo_objects_3d_data: np.ndarray,
    reprojection_error_fr_mar: np.ndarray,
    reprojection_error_cam_fr_mar: np.ndarray,
    data_folder_path: Path,
):
    """
    Save YOLO objects 3D data to files.
    """
    # Save main 3D data
    output_path = data_folder_path / YOLO_OBJECT_MARKERS_3D_NPY_FILE_NAME
    np.save(output_path, yolo_objects_3d_data)
    logger.info(f"Saved YOLO objects 3D data to: {output_path}")
    
    # Save reprojection error data
    reprojection_error_path = data_folder_path / "yolo_objects_reprojection_error.npy"
    np.save(reprojection_error_path, reprojection_error_fr_mar)
    
    reprojection_error_cam_path = data_folder_path / "yolo_objects_reprojection_error_cam.npy"
    np.save(reprojection_error_cam_path, reprojection_error_cam_fr_mar)


