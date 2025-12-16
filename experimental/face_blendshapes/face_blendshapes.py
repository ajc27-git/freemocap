import logging
import time
import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)

def start_face_blendshapes():
    logger.debug("Starting Face Blendshapes!")

    # skellytracker_blendshapes -i "C:\Users\andre\freemocap_data\recording_sessions\freemocap_sample_data\facecam\20241004_facecam.mp4" -o "C:\Users\andre\freemocap_data\recording_sessions\freemocap_sample_data\facecam\20241004_facecam_annotated.mp4" -c "C:\Users\andre\freemocap_data\recording_sessions\freemocap_sample_data\facecam\20241004_facecam_blendshapes.csv"


if __name__ == "__main__":
    start_face_blendshapes()

