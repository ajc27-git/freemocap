import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from freemocap.core_processes.capture_volume_calibration.save_3d_data_to_npy import (
    save_3d_data_to_npy,
)
from freemocap.core_processes.capture_volume_calibration.triangulate_3d_data import triangulate_3d_data
from freemocap.data_layer.recording_models.post_processing_parameter_models import ProcessingParameterModel

logger = logging.getLogger(__name__)


def run_reprojection_error_filtering(
    image_data_numCams_numFrames_numTrackedPts_XYZ: np.ndarray,
    raw_skel3d_frame_marker_xyz: np.ndarray,
    skeleton_reprojection_error_cam_fr_mar: np.ndarray,
    skeleton_reprojection_error_fr_mar: np.ndarray,
    anipose_calibration_object,
    processing_parameters: ProcessingParameterModel,
) -> np.ndarray:
    """
    Runs reprojection error filtering on 3d data, saves the filtered 3d data and reprojection error data, and creates and saves a debug plot.

    :param image_data_numCams_numFrames_numTrackedPts_XYZ: original 2d data
    :param raw_skel3d_frame_marker_xyz: original 3d data
    :param skeleton_reprojection_error_cam_fr_mar: original per camerareprojection error
    :param skeleton_reprojection_error_fr_mar: original per frame average reprojection error
    :param anipose_calibration_object: anipose calibration object
    :param processing_parameters: processing parameters

    :return: filtered 3d data
    """
    if hasattr(
        processing_parameters.tracking_model_info, "num_tracked_points_body"
    ):  # we don't want to reproject hand and face data
        num_tracked_points = processing_parameters.tracking_model_info.num_tracked_points_body
    else:
        num_tracked_points = processing_parameters.tracking_model_info.num_tracked_points

    # (
    #     reprojection_filtered_skel3d_frame_marker_xyz,
    #     reprojection_filtered_skeleton_reprojection_error_fr_mar,
    #     reprojection_filtered_skeleton_reprojection_error_cam_fr_mar,
    # ) = filter_by_reprojection_error(
    #     reprojection_error_camera_frame_marker=skeleton_reprojection_error_cam_fr_mar,
    #     reprojection_error_frame_marker=skeleton_reprojection_error_fr_mar,
    #     reprojection_error_confidence_threshold=processing_parameters.anipose_triangulate_3d_parameters_model.reprojection_error_confidence_cutoff,
    #     image_2d_data=image_data_numCams_numFrames_numTrackedPts_XYZ[:, :, :, :2],
    #     raw_skel3d_frame_marker_xyz=raw_skel3d_frame_marker_xyz,
    #     anipose_calibration_object=anipose_calibration_object,
    #     num_tracked_points=num_tracked_points,
    #     use_triangulate_ransac=processing_parameters.anipose_triangulate_3d_parameters_model.use_triangulate_ransac_method,
    #     minimum_cameras_to_reproject=processing_parameters.anipose_triangulate_3d_parameters_model.minimum_cameras_to_reproject,
    # )

    (
        reprojection_filtered_skel3d_frame_marker_xyz,
        reprojection_filtered_skeleton_reprojection_error_fr_mar,
        reprojection_filtered_skeleton_reprojection_error_cam_fr_mar,
    ) = filter_by_reprojection_error_iterative(
        image_2d_data=image_data_numCams_numFrames_numTrackedPts_XYZ[:, :, :, :2],
        raw_skel3d_frame_marker_xyz=raw_skel3d_frame_marker_xyz,
        anipose_calibration_object=anipose_calibration_object,
        reprojection_error_frame_marker=skeleton_reprojection_error_fr_mar,
        reprojection_error_camera_frame_marker=skeleton_reprojection_error_cam_fr_mar,
        minimum_cameras_to_reproject=3,
        error_multiplier_threshold=2.5,
        max_iterations=5,
    )

    save_3d_data_to_npy(
        data3d_numFrames_numTrackedPoints_XYZ=reprojection_filtered_skel3d_frame_marker_xyz,
        data3d_numFrames_numTrackedPoints_reprojectionError=reprojection_filtered_skeleton_reprojection_error_fr_mar,
        data3d_numCams_numFrames_numTrackedPoints_reprojectionError=reprojection_filtered_skeleton_reprojection_error_cam_fr_mar,
        path_to_folder_where_data_will_be_saved=processing_parameters.recording_info_model.raw_data_folder_path,
        processing_level="reprojection_filtered",
        file_prefix=processing_parameters.tracking_model_info.name,
    )
    plot_reprojection_error(
        raw_reprojection_error_frame_marker=skeleton_reprojection_error_fr_mar,
        filtered_reprojection_error_frame_marker=reprojection_filtered_skeleton_reprojection_error_fr_mar,
        reprojection_error_threshold=float(
            np.nanpercentile(
                skeleton_reprojection_error_cam_fr_mar,
                processing_parameters.anipose_triangulate_3d_parameters_model.reprojection_error_confidence_cutoff,
            )
        ),
        output_folder_path=processing_parameters.recording_info_model.raw_data_folder_path,
    )
    return reprojection_filtered_skel3d_frame_marker_xyz


def filter_by_reprojection_error(
    reprojection_error_camera_frame_marker: np.ndarray,
    reprojection_error_frame_marker: np.ndarray,
    reprojection_error_confidence_threshold: float,
    image_2d_data: np.ndarray,
    raw_skel3d_frame_marker_xyz: np.ndarray,
    anipose_calibration_object,
    num_tracked_points: int,
    use_triangulate_ransac: bool = False,
    minimum_cameras_to_reproject: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_cameras = image_2d_data.shape[0]
    num_cameras_to_remove = 1

    if total_cameras <= minimum_cameras_to_reproject:
        logger.warning(
            f"Not enough cameras to filter by reprojection error. There are {total_cameras} cameras, but minimum number of cameras is {minimum_cameras_to_reproject}. Returning unfiltered data."
        )
        return (
            raw_skel3d_frame_marker_xyz,
            reprojection_error_frame_marker,
            reprojection_error_camera_frame_marker,
        )

    body2d_camera_frame_marker_xy = image_2d_data[:, :, :num_tracked_points, :2]
    bodyReprojErr_camera_frame_marker = reprojection_error_camera_frame_marker[:, :, :num_tracked_points]

    reprojection_error_confidence_threshold = min(max(reprojection_error_confidence_threshold, 0), 100)

    reprojection_error_threshold = np.nanpercentile(
        np.nanmean(bodyReprojErr_camera_frame_marker, axis=0), reprojection_error_confidence_threshold, method="weibull"
    )  # TODO: try running this on reprojection_error_frame_marker with body points pulled out, rather than with cameras included
    logger.info(f"Using reprojection error threshold of {reprojection_error_threshold}")

    data_to_reproject_camera_frame_marker_xy, unique_frame_marker_list = _get_data_to_reproject(
        num_cameras_to_remove=num_cameras_to_remove,
        reprojection_error_threshold=reprojection_error_threshold,
        reprojError_cam_frame_marker=bodyReprojErr_camera_frame_marker,
        input_2d_data_camera_frame_marker_xy=body2d_camera_frame_marker_xy,
    )

    while len(unique_frame_marker_list) > 0 and total_cameras - num_cameras_to_remove >= minimum_cameras_to_reproject:
        logger.info("Retriangulating data")
        (
            retriangulated_data_frame_marker_xyz,
            new_reprojection_error_flat,
            new_reprojError_cam_frame_marker,
        ) = triangulate_3d_data(
            anipose_calibration_object=anipose_calibration_object,
            image_2d_data=data_to_reproject_camera_frame_marker_xy,
            use_triangulate_ransac=use_triangulate_ransac,
        )

        num_cameras_to_remove += 1

        data_to_reproject_camera_frame_marker_xy, unique_frame_marker_list = _get_data_to_reproject(
            num_cameras_to_remove=num_cameras_to_remove,
            reprojection_error_threshold=reprojection_error_threshold,
            reprojError_cam_frame_marker=new_reprojError_cam_frame_marker,
            input_2d_data_camera_frame_marker_xy=data_to_reproject_camera_frame_marker_xy,
        )

    # nan remaining data above threshold
    _nan_data_above_threshold(unique_frame_marker_list, retriangulated_data_frame_marker_xyz)

    # put retriangulated data back in place
    filtered_skel3d_frame_marker_xyz = raw_skel3d_frame_marker_xyz.copy()
    filtered_skel3d_frame_marker_xyz[:, :num_tracked_points, :] = retriangulated_data_frame_marker_xyz

    filtered_reprojection_error_frame_marker = reprojection_error_frame_marker.copy()
    filtered_reprojection_error_frame_marker[:, :num_tracked_points] = new_reprojection_error_flat

    filtered_reprojection_error_camera_frame_marker = reprojection_error_camera_frame_marker.copy()
    filtered_reprojection_error_camera_frame_marker[:, :, :num_tracked_points] = new_reprojError_cam_frame_marker

    return (
        filtered_skel3d_frame_marker_xyz,
        filtered_reprojection_error_frame_marker,
        filtered_reprojection_error_camera_frame_marker,
    )


def _nan_data_above_threshold(unique_frame_marker_list: list, retriangulated_data_frame_marker_xyz: np.ndarray) -> None:
    if len(unique_frame_marker_list) > 0:
        logger.info(
            f"Out of cameras to remove, setting {len(unique_frame_marker_list)} points with reprojection error above threshold to NaNs"
        )
        for frame_marker in unique_frame_marker_list:
            retriangulated_data_frame_marker_xyz[frame_marker[0], frame_marker[1], :] = np.nan


def _get_data_to_reproject(
    num_cameras_to_remove: int,
    reprojection_error_threshold: float,
    reprojError_cam_frame_marker: np.ndarray,
    input_2d_data_camera_frame_marker_xy: np.ndarray,
) -> tuple[np.ndarray, list]:
    indices_above_threshold = np.nonzero(reprojError_cam_frame_marker > reprojection_error_threshold)
    logger.debug(f"SHAPE OF INDICES ABOVE THRESHOLD: {indices_above_threshold[0].shape}")

    total_frame_marker_combos = (
        input_2d_data_camera_frame_marker_xy.shape[1] * input_2d_data_camera_frame_marker_xy.shape[2]
    )
    unique_frame_marker_list = _get_unique_frame_marker_list(indices_above_threshold=indices_above_threshold)
    logger.info(
        f"number of frame/marker combos with reprojection error above threshold: {len(unique_frame_marker_list)} ({len(unique_frame_marker_list) / total_frame_marker_combos * 100:.1f} percent of total)"
    )

    cameras_to_remove, frames_to_reproject, markers_to_reproject = _get_camera_frame_marker_lists_to_reproject(
        reprojError_cam_frame_marker=reprojError_cam_frame_marker,
        frame_marker_list=unique_frame_marker_list,
        num_cameras_to_remove=num_cameras_to_remove,
    )

    data_to_reproject_camera_frame_marker_xy = _set_unincluded_data_to_nans(
        input_2d_data=input_2d_data_camera_frame_marker_xy,
        frames_with_reprojection_error=frames_to_reproject,
        markers_with_reprojection_error=markers_to_reproject,
        cameras_to_remove=cameras_to_remove,
    )

    return (data_to_reproject_camera_frame_marker_xy, unique_frame_marker_list)


def _get_unique_frame_marker_list(
    indices_above_threshold: np.ndarray,
) -> list:
    return list(set(zip(indices_above_threshold[1], indices_above_threshold[2])))


def _get_camera_frame_marker_lists_to_reproject(
    reprojError_cam_frame_marker: np.ndarray,
    frame_marker_list: list,
    num_cameras_to_remove: int,
) -> Tuple[list, list, list]:
    """
    Generate the lists of cameras, frames, and markers to reproject based on the given input.
    Find the cameras with the worst reprojection errors for the given frames and markers.
    Args:
        reprojError_cam_frame_marker (np.ndarray): The array containing the reprojection errors for each camera, frame, and marker.
        frame_marker_list (list): The list of tuples containing the frame and marker indices.
        num_cameras_to_remove (int): The number of cameras to remove.
    Returns:
        Tuple[list, list, list]: A tuple containing the lists of cameras to remove, frames to reproject, and markers to reproject.
    """
    cameras_to_remove = []
    frames_to_reproject = []
    markers_to_reproject = []
    for frame, marker in frame_marker_list:
        frames_to_reproject.append(frame)
        markers_to_reproject.append(marker)
        max_indices = reprojError_cam_frame_marker[:, frame, marker].argsort()[::-1][:num_cameras_to_remove]
        cameras_to_remove.append(list(max_indices))
    return (cameras_to_remove, frames_to_reproject, markers_to_reproject)


def _set_unincluded_data_to_nans(
    input_2d_data: np.ndarray,
    frames_with_reprojection_error: list,
    markers_with_reprojection_error: list,
    cameras_to_remove: list[list[int]],
) -> np.ndarray:
    data_to_reproject = input_2d_data.copy()
    for list_of_cameras, frame, marker in zip(
        cameras_to_remove, frames_with_reprojection_error, markers_with_reprojection_error
    ):
        for camera in list_of_cameras:
            data_to_reproject[camera, frame, marker, :] = np.nan
    return data_to_reproject


def plot_reprojection_error(
    raw_reprojection_error_frame_marker: np.ndarray,
    filtered_reprojection_error_frame_marker: np.ndarray,
    reprojection_error_threshold: float,
    output_folder_path: Union[str, Path],
) -> None:
    title = "Mean Reprojection Error Per Frame"
    file_name = "debug_reprojection_error_filtering.png"
    output_filepath = Path(output_folder_path) / file_name
    raw_mean_reprojection_error_per_frame = np.nanmean(
        raw_reprojection_error_frame_marker,
        axis=1,
    )
    filtered_mean_reprojection_error_per_frame = np.nanmean(
        filtered_reprojection_error_frame_marker,
        axis=1,
    )
    plt.plot(raw_mean_reprojection_error_per_frame, color="blue", label="Data Before Filtering")
    plt.plot(filtered_mean_reprojection_error_per_frame, color="orange", alpha=0.9, label="Data After Filtering")
    plt.xlabel("Frame")
    plt.ylabel("Mean Reprojection Error Across Markers (mm)")
    plt.ylim(0, 2 * reprojection_error_threshold)
    plt.hlines(
        y=reprojection_error_threshold,
        xmin=0,
        xmax=len(raw_mean_reprojection_error_per_frame),
        color="red",
        label="Cutoff Threshold",
    )
    plt.title(title)
    plt.legend(loc="upper right")
    logger.info(f"Saving debug plots to: {output_filepath}")
    plt.savefig(output_filepath, dpi=300)


import numpy as np
import logging

logger = logging.getLogger(__name__)

def filter_by_reprojection_error_iterative(
    image_2d_data: np.ndarray,
    raw_skel3d_frame_marker_xyz: np.ndarray,
    anipose_calibration_object,
    reprojection_error_frame_marker: np.ndarray,
    reprojection_error_camera_frame_marker: np.ndarray,
    minimum_cameras_to_reproject: int = 2,
    error_multiplier_threshold: float = 2.5,
    max_iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative per-frame, per-marker reprojection filtering.
    """
    num_cams, num_frames, num_markers, _ = image_2d_data.shape
    filtered_skel3d_frame_marker_xyz = raw_skel3d_frame_marker_xyz.copy()
    
    # Inicializar arrays de error actualizados
    updated_reprojection_error_frame_marker = reprojection_error_frame_marker.copy()
    updated_reprojection_error_camera_frame_marker = reprojection_error_camera_frame_marker.copy()

    logger.info(f"Executing filtering on {num_frames} frames, {num_markers} markers")
    logger.info(f"Parameters: minimum_cameras_to_reproject={minimum_cameras_to_reproject}, error_multiplier_threshold={error_multiplier_threshold}, max_iterations={max_iterations}")

    # Define problematic frames range for detailed debugging
    PROBLEMATIC_FRAMES = range(47, 48)
    # TARGET_MARKERS = range(0, 75)  # Body and hands
    TARGET_MARKERS = range(27, 28)  # Body

    # --- Bucle principal: frame por frame, marker por marker ---
    points_processed = 0
    points_filtered = 0
    cameras_removed_total = 0
    
    for frame in range(num_frames):
        # Only process problematic frames for debugging
        if frame not in PROBLEMATIC_FRAMES:
            # Skip non-problematic frames or copy original data
            continue
            
        logger.info(f"=== PROCESSING PROBLEMATIC FRAME {frame} ===")
            
        for marker in range(num_markers):
            # Only process foot markers for debugging
            if marker not in TARGET_MARKERS:
                continue
                
            points_processed += 1
            
            # Find valid cameras for this point (non-NaN)
            valid_cams = []
            for cam_idx in range(num_cams):
                if not np.any(np.isnan(image_2d_data[cam_idx, frame, marker, :])):
                    valid_cams.append(cam_idx)
            
            logger.info(f"Frame {frame}, Marker {marker}: {len(valid_cams)} valid cameras: {valid_cams}")
            
            # If not enough cameras, mark as NaN and continue
            if len(valid_cams) < minimum_cameras_to_reproject:
                logger.warning(f"Frame {frame}, Marker {marker}: Insufficient cameras ({len(valid_cams)}), setting to NaN")
                filtered_skel3d_frame_marker_xyz[frame, marker, :] = np.nan
                points_filtered += 1
                continue

            # List of cameras to use (start with all valid ones)
            cams_to_use = valid_cams.copy()
            best_point_3d = np.full(3, np.nan)
            best_reprojection_errors = np.full(num_cams, np.nan)
            
            # Store original 2D points for comparison
            original_2d_points = {}
            for cam_idx in valid_cams:
                original_2d_points[cam_idx] = image_2d_data[cam_idx, frame, marker, :].copy()
            
            logger.info(f"Frame {frame}, Marker {marker}: Original 2D points: {original_2d_points}")
            
            # --- Iterations to remove cameras with bad error ---
            for iteration in range(max_iterations):
                logger.info(f"Frame {frame}, Marker {marker}: Iteration {iteration + 1}, using cameras: {cams_to_use}")
                
                # Triangulation with current cameras
                points_2d_subset = image_2d_data[cams_to_use, frame, marker, :]
                
                # Triangulation using anipose
                point_3d, reproj_errors = _triangulate_single_point_anipose(
                    anipose_calibration_object, 
                    points_2d_subset, 
                    cams_to_use
                )
                
                # If triangulation failed, exit
                if np.any(np.isnan(point_3d)):
                    logger.warning(f"Frame {frame}, Marker {marker}: Triangulation failed, resulting in NaN")
                    break
                
                logger.info(f"Frame {frame}, Marker {marker}: 3D point: {point_3d}")
                
                # Calculate reprojection errors for used cameras
                errors_used = []
                cam_errors_map = {}
                for i, cam_idx in enumerate(cams_to_use):
                    if not np.isnan(reproj_errors[i]):
                        errors_used.append(reproj_errors[i])
                        cam_errors_map[cam_idx] = reproj_errors[i]
                        logger.info(f"  Camera {cam_idx}: error = {reproj_errors[i]:.3f}")
                
                # If no valid errors, exit
                if not errors_used:
                    logger.warning(f"Frame {frame}, Marker {marker}: No valid reprojection errors")
                    break
                
                # Calculate error statistics
                mean_error = np.mean(errors_used)
                std_error = np.std(errors_used)
                threshold_error = mean_error * error_multiplier_threshold
                
                logger.info(f"Frame {frame}, Marker {marker}: Error stats - mean={mean_error:.3f}, std={std_error:.3f}, threshold={threshold_error:.3f}")
                
                # Find camera with worst error
                worst_cam = None
                worst_error = -1
                for cam_idx, error in cam_errors_map.items():
                    if error > worst_error:
                        worst_error = error
                        worst_cam = cam_idx
                
                logger.info(f"Frame {frame}, Marker {marker}: Worst camera = {worst_cam} with error {worst_error:.3f}")
                
                # Check if we should remove the worst camera
                if worst_error > threshold_error and len(cams_to_use) > minimum_cameras_to_reproject:
                    # Remove worst camera and continue
                    cams_to_use.remove(worst_cam)
                    cameras_removed_total += 1
                    logger.info(f"Frame {frame}, Marker {marker}: REMOVING camera {worst_cam} (error={worst_error:.3f}, threshold={threshold_error:.3f})")
                    
                    # Compare original vs reprojected points for removed camera
                    reprojected_2d = _get_reprojected_point(anipose_calibration_object, point_3d, worst_cam)
                    if reprojected_2d is not None:
                        original_point = original_2d_points[worst_cam]
                        distance = np.linalg.norm(original_point - reprojected_2d)
                        logger.info(f"  Camera {worst_cam}: Original 2D = {original_point}, Reprojected 2D = {reprojected_2d}, Distance = {distance:.3f}")
                    
                    continue
                else:
                    # All cameras are within threshold or we can't remove more
                    logger.info(f"Frame {frame}, Marker {marker}: All cameras within threshold or minimum reached")
                    best_point_3d = point_3d
                    
                    # Update reprojection errors for all cameras
                    for cam_idx in range(num_cams):
                        if cam_idx in cam_errors_map:
                            best_reprojection_errors[cam_idx] = cam_errors_map[cam_idx]
                        else:
                            best_reprojection_errors[cam_idx] = np.nan
                    break
            else:
                # If we reached max iterations, use the last calculated point
                if not np.any(np.isnan(point_3d)):
                    logger.info(f"Frame {frame}, Marker {marker}: Using point from max iterations")
                    best_point_3d = point_3d
                    for i, cam_idx in enumerate(cams_to_use):
                        if not np.isnan(reproj_errors[i]):
                            best_reprojection_errors[cam_idx] = reproj_errors[i]

            # Assign final result
            filtered_skel3d_frame_marker_xyz[frame, marker, :] = best_point_3d
            filtered_skel3d_frame_marker_xyz[frame, marker, 2] += 500  # Direct Z modification
            logger.info(f"Added 500mm to Z coordinate")
            
            # Compare with original 3D point
            original_3d = raw_skel3d_frame_marker_xyz[frame, marker, :]
            if not np.any(np.isnan(original_3d)) and not np.any(np.isnan(best_point_3d)):
                distance_3d = np.linalg.norm(original_3d - best_point_3d)
                logger.info(f"Frame {frame}, Marker {marker}: 3D change - Original: {original_3d}, Filtered: {best_point_3d}, Distance: {distance_3d:.3f}")
            
            # Update error arrays
            if not np.any(np.isnan(best_point_3d)):
                # Average error for this frame/marker
                valid_errors = [err for err in best_reprojection_errors if not np.isnan(err)]
                if valid_errors:
                    updated_reprojection_error_frame_marker[frame, marker] = np.mean(valid_errors)
                
                # Errors per camera
                for cam_idx in range(num_cams):
                    if not np.isnan(best_reprojection_errors[cam_idx]):
                        updated_reprojection_error_camera_frame_marker[cam_idx, frame, marker] = best_reprojection_errors[cam_idx]
            else:
                points_filtered += 1
                logger.warning(f"Frame {frame}, Marker {marker}: Final result is NaN")

    # Log results
    logger.info(f"Iterative filtering completed:")
    logger.info(f"  - Processed Points: {points_processed}")
    logger.info(f"  - Filtered Points (NaN): {points_filtered} ({points_filtered/points_processed*100:.1f}%)")
    logger.info(f"  - Total cameras removed: {cameras_removed_total}")
    
    # NaN statistics after filtering
    total_nans_after = np.isnan(filtered_skel3d_frame_marker_xyz[:, :, 0]).sum()
    logger.info(f"  - NaNs after filtering: {total_nans_after} ({(total_nans_after/(num_frames*num_markers))*100:.1f}%)")
    
    return (
        filtered_skel3d_frame_marker_xyz,
        updated_reprojection_error_frame_marker,
        updated_reprojection_error_camera_frame_marker,
    )


def _triangulate_single_point_anipose(calibration_object, points_2d_subset, cam_indices):
    """
    Triangulates a single point using the Anipose calibration object.
    points_2d_subset: array of shape (n_cams_used, 2) containing the 2D points
    cam_indices: list of indices of the cameras used
    """
    n_cams_used = len(cam_indices)
    
    # Create full array with NaN for all cameras
    points_2d_full = np.full((len(calibration_object.cameras), 1, 2), np.nan)
    
    # Put valid points in their corresponding positions
    for i, cam_idx in enumerate(cam_indices):
        points_2d_full[cam_idx, 0, :] = points_2d_subset[i]
    
    # Simple triangulation
    try:
        point_3d = calibration_object.triangulate(points_2d_full, undistort=True)
        
        # Calculate reprojection errors
        reproj_errors_full = calibration_object.reprojection_error(
            point_3d.reshape(1, 3), 
            points_2d_full, 
            mean=False
        )
        
        # Extract errors only for used cameras
        reproj_errors = np.full(n_cams_used, np.nan)
        for i, cam_idx in enumerate(cam_indices):
            error_vector = reproj_errors_full[cam_idx, 0, :]
            if not np.any(np.isnan(error_vector)):
                reproj_errors[i] = np.linalg.norm(error_vector)
        
        return point_3d, reproj_errors
        
    except Exception as e:
        logger.debug(f"Triangulation error: {e}")
        return np.full(3, np.nan), np.full(n_cams_used, np.nan)


def _get_reprojected_point(calibration_object, point_3d, camera_index):
    """
    Get the reprojected 2D point for a specific camera.
    """
    try:
        # Project 3D point to 2D for the specified camera
        point_3d_reshaped = point_3d.reshape(1, 1, 3)
        reprojected = calibration_object.cameras[camera_index].project(point_3d_reshaped)
        return reprojected.reshape(2)
    except Exception as e:
        logger.debug(f"Reprojection error for camera {camera_index}: {e}")
        return None