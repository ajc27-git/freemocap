import logging
from typing import List, Tuple

from pydantic import BaseModel, ConfigDict
from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams
from skellytracker.trackers.base_tracker.model_info import ModelInfo
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import MediapipeTrackingParams, MediapipeModelInfo

from freemocap.data_layer.recording_models.recording_info_model import (
    RecordingInfoModel,
)

logger = logging.getLogger(__name__)


class AniposeTriangulate3DParametersModel(BaseModel):
    run_reprojection_error_filtering: bool = False
    reprojection_error_confidence_cutoff: float = 90
    minimum_cameras_to_reproject: int = 3
    use_triangulate_ransac_method: bool = False
    run_3d_triangulation: bool = True
    flatten_single_camera_data: bool = True


class ButterworthFilterParametersModel(BaseModel):
    sampling_rate: float = 30
    cutoff_frequency: float = 7
    order: int = 4


class PostProcessingParametersModel(BaseModel):
    framerate: float = 30.0
    butterworth_filter_parameters: ButterworthFilterParametersModel = ButterworthFilterParametersModel()
    max_gap_to_fill: int = 10
    run_butterworth_filter: bool = True


class ColorMarkerConfig(BaseModel):
    """Configuration for a single color marker."""
    enabled: bool = False
    target_color_bgr: Tuple[int, int, int] = (0, 0, 255)  # Default: Red in BGR
    color_tolerance: int = 30
    marker_name: str = "color_marker"
    min_contour_area: int = 100


class ColorTrackerParametersModel(BaseModel):
    """Parameters for color-based marker tracking."""
    run_color_tracking: bool = False
    marker_configs: List[ColorMarkerConfig] = [
        ColorMarkerConfig(
            enabled=True,
            target_color_bgr=(0, 0, 255),  # Red in BGR
            color_tolerance=30,
            marker_name="red_marker",
            min_contour_area=100,
        ),
        ColorMarkerConfig(
            enabled=True,
            target_color_bgr=(0, 255, 0),  # Green in BGR
            color_tolerance=25,
            marker_name="green_marker",
            min_contour_area=80,
        ),
        ColorMarkerConfig(
            enabled=False,
            target_color_bgr=(255, 0, 0),  # Blue in BGR
            color_tolerance=30,
            marker_name="blue_marker",
            min_contour_area=100,
        ),
    ]
    use_morphological_ops: bool = True
    num_processes: int = 1  # Color tracking is lightweight, usually 1 process is enough


class ProcessingParameterModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    recording_info_model: RecordingInfoModel = None
    tracking_parameters_model: BaseTrackingParams = MediapipeTrackingParams()
    color_tracker_parameters_model: ColorTrackerParametersModel = ColorTrackerParametersModel()
    anipose_triangulate_3d_parameters_model: AniposeTriangulate3DParametersModel = AniposeTriangulate3DParametersModel()
    post_processing_parameters_model: PostProcessingParametersModel = PostProcessingParametersModel()
    tracking_model_info: ModelInfo = MediapipeModelInfo()
