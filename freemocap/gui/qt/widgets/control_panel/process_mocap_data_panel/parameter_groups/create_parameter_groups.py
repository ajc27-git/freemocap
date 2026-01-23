from pyqtgraph.parametertree import Parameter
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import (
    MediapipeTrackingParams,
)

from freemocap.data_layer.recording_models.post_processing_parameter_models import (
    ProcessingParameterModel,
    AniposeTriangulate3DParametersModel,
    PostProcessingParametersModel,
    ButterworthFilterParametersModel,
    ColorMarkerConfig,
    ColorTrackerParametersModel,
)

import logging

BUTTERWORTH_ORDER = "Order"

BUTTERWORTH_CUTOFF_FREQUENCY = "Cutoff Frequency"

POST_PROCESSING_FRAME_RATE = "Framerate"

BUTTERWORTH_FILTER_TREE_NAME = "Butterworth Filter"

USE_RANSAC_METHOD = "Use RANSAC Method"

ANIPOSE_CONFIDENCE_CUTOFF = "Confidence Threshold Cut-off"

REPROJECTION_ERROR_FILTERING_TREE_NAME = "Reprojection Error Filtering"

RUN_REPROJECTION_ERROR_FILTERING = "Run Reprojection Error Filtering"

REPROJECTION_ERROR_FILTER_THRESHOLD = "Reprojection Error Filter Threshold (%)"

MINIMUM_CAMERAS_TO_REPROJECT = "Minimum Cameras to Reproject"

FLATTEN_SINGLE_CAMERA_DATA = "Flatten Single Camera Data (Recommended)"

ANIPOSE_TREE_NAME = "Anipose Triangulation"

YOLO_CROP_TREE_NAME = "YOLO Crop"

USE_YOLO_CROP_METHOD = "Use YOLO Crop Method"

YOLO_MODEL_SIZE = "YOLO Model Size"

BOUNDING_BOX_BUFFER_METHOD = "Buffer Bounding Box:"

BOUNDING_BOX_BUFFER_PERCENTAGE = "Bounding Box Buffer Percentage"

STATIC_IMAGE_MODE = "Static Image Mode"

MINIUMUM_TRACKING_CONFIDENCE = "Minimum Tracking Confidence"

MINIMUM_DETECTION_CONFIDENCE = "Minimum Detection Confidence"

MEDIAPIPE_MODEL_COMPLEXITY = "Model Complexity"

MEDIAPIPE_TREE_NAME = "Mediapipe"

RUN_IMAGE_TRACKING_NAME = "Run 2d image tracking?"

COLOR_TRACKER_TREE_NAME = "Color Tracker"

RUN_COLOR_TRACKER_NAME = "Run color tracker?"

RUN_3D_TRIANGULATION_NAME = "Run 3d triangulation?"

RUN_BUTTERWORTH_FILTER_NAME = "Run butterworth filter?"

NUMBER_OF_PROCESSES_PARAMETER_NAME = "Max Number of Processes to Use"

# Color Marker 1 constants
MARKER_1_ENABLED = "Marker 1 - Enabled"
MARKER_1_COLOR = "Marker 1 - Color"
MARKER_1_HUE_TOLERANCE = "Marker 1 - Hue Tolerance (0-179)"
MARKER_1_SATURATION_TOLERANCE = "Marker 1 - Saturation Tolerance (0-255)"
MARKER_1_VALUE_TOLERANCE = "Marker 1 - Value Tolerance (0-255)"
MARKER_1_MIN_AREA = "Marker 1 - Minimum Area"
MARKER_1_NAME = "Marker 1 - Name"

# Color Marker 2 constants
MARKER_2_ENABLED = "Marker 2 - Enabled"
MARKER_2_COLOR = "Marker 2 - Color"
MARKER_2_HUE_TOLERANCE = "Marker 2 - Hue Tolerance (0-179)"
MARKER_2_SATURATION_TOLERANCE = "Marker 2 - Saturation Tolerance (0-255)"
MARKER_2_VALUE_TOLERANCE = "Marker 2 - Value Tolerance (0-255)"
MARKER_2_MIN_AREA = "Marker 2 - Minimum Area"
MARKER_2_NAME = "Marker 2 - Name"

# Color Marker 3 constants
MARKER_3_ENABLED = "Marker 3 - Enabled"
MARKER_3_COLOR = "Marker 3 - Color"
MARKER_3_HUE_TOLERANCE = "Marker 3 - Hue Tolerance (0-179)"
MARKER_3_SATURATION_TOLERANCE = "Marker 3 - Saturation Tolerance (0-255)"
MARKER_3_VALUE_TOLERANCE = "Marker 3 - Value Tolerance (0-255)"
MARKER_3_MIN_AREA = "Marker 3 - Minimum Area"
MARKER_3_NAME = "Marker 3 - Name"


# TODO: figure out how to generalize this
def create_mediapipe_parameter_group(
        parameter_model: MediapipeTrackingParams,
) -> Parameter:
    mediapipe_model_complexity_list = [
        "0 (Fastest/Least accurate)",
        "1 (Middle ground)",
        "2 (Slowest/Most accurate)",
    ]
    return Parameter.create(
        name=MEDIAPIPE_TREE_NAME,
        type="group",
        children=[
            dict(
                name=YOLO_CROP_TREE_NAME,
                type="group",
                children=[
                    dict(
                        name=USE_YOLO_CROP_METHOD,
                        type="bool",
                        value=parameter_model.use_yolo_crop_method,
                        tip="If true, `skellytracker` will use YOLO to pre-crop the person from the image before running the `mediapipe` tracker",
                    ),
                    dict(
                        name=YOLO_MODEL_SIZE,
                        type="list",
                        limits=["nano", "small", "medium", "large", "extra_large", "high_res"],
                        value=parameter_model.yolo_model_size,
                        tip="Smaller models are faster but may be less accurate",
                    ),
                    dict(
                        name=BOUNDING_BOX_BUFFER_METHOD,
                        type="list",
                        limits=["By box size", "By image size"],
                        value=parameter_model.buffer_size_method,
                        tip="Buffer bounding box by percentage of either box size or image size",
                    ),
                    dict(
                        name=BOUNDING_BOX_BUFFER_PERCENTAGE,
                        type="int",
                        value=parameter_model.bounding_box_buffer_percentage,
                        limits=(0, 100),
                        step=1,
                        tip="Percentage to increase size of bounding box",
                    ),
                ],
            ),
            dict(
                name=MEDIAPIPE_MODEL_COMPLEXITY,
                type="list",
                limits=mediapipe_model_complexity_list,
                value=mediapipe_model_complexity_list[parameter_model.mediapipe_model_complexity],
                tip="Which Mediapipe model to use - higher complexity is slower but more accurate. "
                    "Variable name in `mediapipe` code: `mediapipe_model_complexity`",
            ),
            dict(
                name=MINIMUM_DETECTION_CONFIDENCE,
                type="float",
                value=parameter_model.min_detection_confidence,
                step=0.05,
                limits=(0.0, 1.0),
                tip="Minimum confidence for a skeleton detection to be considered valid. "
                    "Variable name in `mediapipe` code: `min_detection_confidence`."
                    "NOTE - Never trust a machine learning model's estimates of their own confidence!",
            ),
            dict(
                name=MINIUMUM_TRACKING_CONFIDENCE,
                type="float",
                value=parameter_model.min_tracking_confidence,
                step=0.05,
                limits=(0.0, 1.0),
                tip="Minimum confidence needed to use the previous frame's skeleton estiamte to predict the next one"
                    "Variable name in `mediapipe` code: `min_tracking_confidence`.",
            ),
            dict(
                name=STATIC_IMAGE_MODE,
                type="bool",
                value=parameter_model.static_image_mode,
                tip="If true, the model will process each image independently, without tracking across frames."
                    "I think this is equivalent to setting `min_tracking_confidence` to 0.0"
                    "Variable name in `mediapipe` code: `static_image_mode`",
            ),
        ],
    )


def create_color_tracker_parameter_group() -> Parameter:
    return Parameter.create(
        name=COLOR_TRACKER_TREE_NAME,
        type="group",
        children=[
            dict(
                name=RUN_COLOR_TRACKER_NAME,
                type="bool",
                value=True,
                tip="If enabled, track colored markers in addition to MediaPipe skeleton",
            ),
            dict(
                name="Marker 1 Settings",
                type="group",
                children=[
                    dict(
                        name=MARKER_1_ENABLED,
                        type="bool",
                        value=True,
                        tip="Enable tracking for Marker 1",
                    ),
                    dict(
                        name=MARKER_1_NAME,
                        type="str",
                        value="red_marker",
                        tip="Name for this marker (used in output files)",
                    ),
                    dict(
                        name=MARKER_1_COLOR,
                        type="color",
                        value=(255, 0, 0),
                        tip="Click to pick color for Marker 1",
                    ),
                    dict(
                        name=MARKER_1_HUE_TOLERANCE,
                        type="int",
                        value=20,
                        limits=(0, 179),
                        tip="Hue tolerance for matching (0-179)",
                    ),
                    dict(
                        name=MARKER_1_SATURATION_TOLERANCE,
                        type="int",
                        value=70,
                        limits=(0, 255),
                        tip="Saturation tolerance for matching (0-255)",
                    ),
                    dict(
                        name=MARKER_1_VALUE_TOLERANCE,
                        type="int",
                        value=70,
                        limits=(0, 255),
                        tip="Color tolerance for matching (higher = more variation allowed)",
                    ),
                    dict(
                        name=MARKER_1_MIN_AREA,
                        type="int",
                        value=10,
                        limits=(10, 10000),
                        tip="Minimum contour area to consider as valid marker",
                    ),
                ],
            ),
            dict(
                name="Marker 2 Settings",
                type="group",
                children=[
                    dict(
                        name=MARKER_2_ENABLED,
                        type="bool",
                        value=True,
                        tip="Enable tracking for Marker 2",
                    ),
                    dict(
                        name=MARKER_2_NAME,
                        type="str",
                        value="green_marker",
                        tip="Name for this marker (used in output files)",
                    ),
                    dict(
                        name=MARKER_2_COLOR,
                        type="color",
                        value=(0, 255, 0),
                        tip="Click to pick color for Marker 2",
                    ),
                    dict(
                        name=MARKER_2_HUE_TOLERANCE,
                        type="int",
                        value=20,
                        limits=(0, 179),
                        tip="Hue tolerance for matching (0-179)",
                    ),
                    dict(
                        name=MARKER_2_SATURATION_TOLERANCE,
                        type="int",
                        value=70,
                        limits=(0, 255),
                        tip="Saturation tolerance for matching (0-255)",
                    ),
                    dict(
                        name=MARKER_2_VALUE_TOLERANCE,
                        type="int",
                        value=70,
                        limits=(0, 255),
                        tip="Color tolerance for matching (higher = more variation allowed)",
                    ),
                    dict(
                        name=MARKER_2_MIN_AREA,
                        type="int",
                        value=10,
                        limits=(10, 10000),
                        tip="Minimum contour area to consider as valid marker",
                    ),
                ],
            ),
            dict(
                name="Marker 3 Settings",
                type="group",
                children=[
                    dict(
                        name=MARKER_3_ENABLED,
                        type="bool",
                        value=True,
                        tip="Enable tracking for Marker 3",
                    ),
                    dict(
                        name=MARKER_3_NAME,
                        type="str",
                        value="blue_marker",
                        tip="Name for this marker (used in output files)",
                    ),
                    dict(
                        name=MARKER_3_COLOR,
                        type="color",
                        value=(0, 0, 255),
                        tip="Click to pick color for Marker 3",
                    ),
                    dict(
                        name=MARKER_3_HUE_TOLERANCE,
                        type="int",
                        value=20,
                        limits=(0, 179),
                        tip="Hue tolerance for matching (0-179)",
                    ),
                    dict(
                        name=MARKER_3_SATURATION_TOLERANCE,
                        type="int",
                        value=70,
                        limits=(0, 255),
                        tip="Saturation tolerance for matching (0-255)",
                    ),
                    dict(
                        name=MARKER_3_VALUE_TOLERANCE,
                        type="int",
                        value=70,
                        limits=(0, 255),
                        tip="Color tolerance for matching (higher = more variation allowed)",
                    ),
                    dict(
                        name=MARKER_3_MIN_AREA,
                        type="int",
                        value=10,
                        limits=(10, 10000),
                        tip="Minimum contour area to consider as valid marker",
                    ),
                ],
            ),
        ],
        tip="Track colored markers using simple color detection (no model training required)",
    )


def create_3d_triangulation_parameter_group(
        parameter_model: AniposeTriangulate3DParametersModel = None,
) -> Parameter:
    if parameter_model is None:
        parameter_model = AniposeTriangulate3DParametersModel()

    return Parameter.create(
        name=ANIPOSE_TREE_NAME,
        type="group",
        children=[
            dict(
                name=USE_RANSAC_METHOD,
                type="bool",
                value=parameter_model.use_triangulate_ransac_method,
                tip="If true, use `anipose`'s `triangulate_ransac` method instead of the default `triangulate_simple` method. "
                    "NOTE - Much slower than the 'simple' method, but might be more accurate and better at rejecting bad camera views. Needs more testing and evaluation to see if it's worth it. ",
            ),
            dict(
                name=FLATTEN_SINGLE_CAMERA_DATA,
                type="bool",
                value=parameter_model.flatten_single_camera_data,
                tip="If true, flatten the data from single camera recordings.",
            ),
            dict(
                name=REPROJECTION_ERROR_FILTERING_TREE_NAME,
                type="group",
                children=[
                    dict(
                        name=RUN_REPROJECTION_ERROR_FILTERING,
                        type="bool",
                        value=parameter_model.run_reprojection_error_filtering,
                        tip="If true, run filtering of reprojection error.",
                    ),
                    dict(
                        name=REPROJECTION_ERROR_FILTER_THRESHOLD,
                        type="float",
                        value=parameter_model.reprojection_error_confidence_cutoff,
                        tip="The maximum reprojection error allowed in the data.",
                    ),
                    dict(
                        name=MINIMUM_CAMERAS_TO_REPROJECT,
                        type="int",
                        value=parameter_model.minimum_cameras_to_reproject,
                        tip="The minimum number of cameras to reproject during retriangulation.",
                    ),
                ],
            ),
        ],
    )


def create_post_processing_parameter_group(
        parameter_model: PostProcessingParametersModel = None,
) -> Parameter:
    if parameter_model is None:
        parameter_model = PostProcessingParametersModel()

    return Parameter.create(
        name=BUTTERWORTH_FILTER_TREE_NAME,
        type="group",
        children=[
            dict(
                name=POST_PROCESSING_FRAME_RATE,
                type="float",
                value=parameter_model.butterworth_filter_parameters.sampling_rate,
                tip="Framerate of the recording " "TODO - Calculate this from the recorded timestamps....",
            ),
            dict(
                name=BUTTERWORTH_CUTOFF_FREQUENCY,
                type="float",
                value=parameter_model.butterworth_filter_parameters.cutoff_frequency,
                tip="Oscillations above this frequency will be filtered from the data. ",
            ),
            dict(
                name=BUTTERWORTH_ORDER,
                type="int",
                value=parameter_model.butterworth_filter_parameters.order,
                tip="Order of the filter."
                    "NOTE - I'm not really sure what this parameter does, but this is what I see in other people's Methods sections so....   lol",
            ),
        ],
        tip="Low-pass, zero-lag, Butterworth filter to remove high frequency oscillations/noise from the data. ",
    )


def extract_parameter_model_from_parameter_tree(
        parameter_object: Parameter,
) -> ProcessingParameterModel:
    parameter_values_dictionary = extract_processing_parameter_model_from_tree(parameter_object=parameter_object)
    
    # Extract color tracker parameters
    marker_configs = []
    
    # Helper function to convert color from QColor or tuple to BGR (OpenCV format)
    def convert_to_bgr(color_value):
        """Convert color from GUI to BGR for OpenCV."""
        logger = logging.getLogger(__name__)
        
        logger.debug(f"DEBUG COLOR CONVERSION - Input: {color_value}, Type: {type(color_value)}")
        
        if hasattr(color_value, 'getRgb'):  # It's a QColor object
            r, g, b, a = color_value.getRgb()
            logger.debug(f"DEBUG COLOR CONVERSION - QColor.getRgb(): R={r}, G={g}, B={b}, A={a}")
        else:  # It's already a tuple
            if len(color_value) == 4:  # RGBA tuple
                r, g, b, a = color_value
            elif len(color_value) == 3:  # RGB tuple
                r, g, b = color_value
                a = 255
            else:
                r, g, b, a = 255, 0, 0, 255  # Default red
            logger.debug(f"DEBUG COLOR CONVERSION - Tuple: R={r}, G={g}, B={b}, A={a}")
        
        bgr = (b, g, r)
        logger.debug(f"DEBUG COLOR CONVERSION - Converted to BGR: {bgr}")
        logger.debug(f"DEBUG COLOR CONVERSION - Expected BGR for:")
        logger.debug(f"  - Red: (0, 0, 255)")
        logger.debug(f"  - Green: (0, 255, 0)")
        logger.debug(f"  - Blue: (255, 0, 0)")
        
        return bgr
    
    # Marker 1
    marker_configs.append(
        ColorMarkerConfig(
            enabled=parameter_values_dictionary.get(MARKER_1_ENABLED, False),
            target_color_bgr=convert_to_bgr(parameter_values_dictionary.get(MARKER_1_COLOR, (255, 0, 0, 255))),
            hue_tolerance=parameter_values_dictionary.get(MARKER_1_HUE_TOLERANCE, 20),
            saturation_tolerance=parameter_values_dictionary.get(MARKER_1_SATURATION_TOLERANCE, 70),
            value_tolerance=parameter_values_dictionary.get(MARKER_1_VALUE_TOLERANCE, 70),
            marker_name=parameter_values_dictionary.get(MARKER_1_NAME, "red_marker"),
            min_contour_area=parameter_values_dictionary.get(MARKER_1_MIN_AREA, 100),
        )
    )
    
    # Marker 2
    marker_configs.append(
        ColorMarkerConfig(
            enabled=parameter_values_dictionary.get(MARKER_2_ENABLED, False),
            target_color_bgr=convert_to_bgr(parameter_values_dictionary.get(MARKER_2_COLOR, (0, 255, 0, 255))),
            hue_tolerance=parameter_values_dictionary.get(MARKER_2_HUE_TOLERANCE, 20),
            saturation_tolerance=parameter_values_dictionary.get(MARKER_2_SATURATION_TOLERANCE, 70),
            value_tolerance=parameter_values_dictionary.get(MARKER_2_VALUE_TOLERANCE, 70),
            marker_name=parameter_values_dictionary.get(MARKER_2_NAME, "green_marker"),
            min_contour_area=parameter_values_dictionary.get(MARKER_2_MIN_AREA, 80),
        )
    )
    
    # Marker 3
    marker_configs.append(
        ColorMarkerConfig(
            enabled=parameter_values_dictionary.get(MARKER_3_ENABLED, False),
            target_color_bgr=convert_to_bgr(parameter_values_dictionary.get(MARKER_3_COLOR, (0, 0, 255, 255))),
            hue_tolerance=parameter_values_dictionary.get(MARKER_3_HUE_TOLERANCE, 20),
            saturation_tolerance=parameter_values_dictionary.get(MARKER_3_SATURATION_TOLERANCE, 70),
            value_tolerance=parameter_values_dictionary.get(MARKER_3_VALUE_TOLERANCE, 70),
            marker_name=parameter_values_dictionary.get(MARKER_3_NAME, "blue_marker"),
            min_contour_area=parameter_values_dictionary.get(MARKER_3_MIN_AREA, 100),
        )
    )

    return ProcessingParameterModel(
        tracking_parameters_model=MediapipeTrackingParams(
            mediapipe_model_complexity=get_integer_from_mediapipe_model_complexity(
                parameter_values_dictionary[MEDIAPIPE_MODEL_COMPLEXITY]
            ),
            min_detection_confidence=parameter_values_dictionary[MINIMUM_DETECTION_CONFIDENCE],
            min_tracking_confidence=parameter_values_dictionary[MINIUMUM_TRACKING_CONFIDENCE],
            static_image_mode=parameter_values_dictionary[STATIC_IMAGE_MODE],
            run_image_tracking=parameter_values_dictionary[RUN_IMAGE_TRACKING_NAME],
            num_processes=parameter_values_dictionary[NUMBER_OF_PROCESSES_PARAMETER_NAME],
            use_yolo_crop_method=parameter_values_dictionary[USE_YOLO_CROP_METHOD],
            yolo_model_size=parameter_values_dictionary[YOLO_MODEL_SIZE],
            buffer_size_method=get_bounding_box_buffer_method_from_string(
                parameter_values_dictionary[BOUNDING_BOX_BUFFER_METHOD]
            ),
            bounding_box_buffer_percentage=parameter_values_dictionary[BOUNDING_BOX_BUFFER_PERCENTAGE],
        ),
        color_tracker_parameters_model=ColorTrackerParametersModel(
            run_color_tracking=parameter_values_dictionary.get(RUN_COLOR_TRACKER_NAME, False),
            marker_configs=marker_configs,
            use_morphological_ops=True,
            num_processes=1,
        ),
        anipose_triangulate_3d_parameters_model=AniposeTriangulate3DParametersModel(
            run_reprojection_error_filtering=parameter_values_dictionary[RUN_REPROJECTION_ERROR_FILTERING],
            reprojection_error_confidence_cutoff=parameter_values_dictionary[REPROJECTION_ERROR_FILTER_THRESHOLD],
            minimum_cameras_to_reproject=parameter_values_dictionary[MINIMUM_CAMERAS_TO_REPROJECT],
            use_triangulate_ransac_method=parameter_values_dictionary[USE_RANSAC_METHOD],
            flatten_single_camera_data=parameter_values_dictionary[FLATTEN_SINGLE_CAMERA_DATA],
            run_3d_triangulation=parameter_values_dictionary[RUN_3D_TRIANGULATION_NAME],
        ),
        post_processing_parameters_model=PostProcessingParametersModel(
            framerate=parameter_values_dictionary[POST_PROCESSING_FRAME_RATE],
            butterworth_filter_parameters=ButterworthFilterParametersModel(
                sampling_rate=parameter_values_dictionary[POST_PROCESSING_FRAME_RATE],
                cutoff_frequency=parameter_values_dictionary[BUTTERWORTH_CUTOFF_FREQUENCY],
                order=parameter_values_dictionary[BUTTERWORTH_ORDER],
            ),
            run_butterworth_filter=parameter_values_dictionary[RUN_BUTTERWORTH_FILTER_NAME],
        ),
    )


def get_integer_from_mediapipe_model_complexity(mediapipe_model_complexity_value: str):
    mediapipe_model_complexity_dictionary = {
        "0 (Fastest/Least accurate)": 0,
        "1 (Middle ground)": 1,
        "2 (Slowest/Most accurate)": 2,
    }
    return mediapipe_model_complexity_dictionary[mediapipe_model_complexity_value]


def get_bounding_box_buffer_method_from_string(buffer_method_string: str) -> str:
    bounding_box_buffer_method_dict = {
        "By box size": "buffer_by_box_size",
        "By image size": "buffer_by_image_size",
    }
    return bounding_box_buffer_method_dict[buffer_method_string]


def extract_processing_parameter_model_from_tree(parameter_object, value_dictionary: dict = None):
    if value_dictionary is None:
        value_dictionary = {}

    for child in parameter_object.children():
        if child.hasChildren():
            extract_processing_parameter_model_from_tree(child, value_dictionary)
        else:
            value_dictionary[child.name()] = child.value()
    return value_dictionary
