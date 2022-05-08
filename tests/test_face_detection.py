import numpy as np
from src.modelling.face_detection.face_detect import face_detecter

# ------------------------------------------------------------------
# Remarks: take note that running pytest from assignment8 folder
#          command: pytest ./tests
# ------------------------------------------------------------------
# ===========
# Positive TC
# ===========
def test_face_detection_load_model():

    model = face_detecter()
    status = model.load_model_and_weight()

    assert status is not None, "model is not loaded properly"

# ===========
# Negative TC
# ===========
def test_face_detection_detect_face_with_invalid_frame_data_type():

    model = face_detecter()
    model.load_model_and_weight()

    frame_data = []
    faces = model.detect_face(frame_data)
    no_of_faces = len(faces)

    if (no_of_faces > 0):
        for (startX, startY, endX, endY) in faces:
            assert (startX == 0 and startY == 0 and endX == 0 and endY == 0), "function unable to detect invalid frame type"
    else:
        assert (no_of_faces == 0), "function unable to detect invalid data type"

def test_face_detection_detect_face_with_invalid_frame_data_shape():

    model = face_detecter()
    model.load_model_and_weight()

    frame_data = np.zeros(shape=(50, 50))
    faces = model.detect_face(frame_data)
    no_of_faces = len(faces)

    if (no_of_faces > 0):
        for (startX, startY, endX, endY) in faces:
            assert (startX == 0 and startY == 0 and endX == 0 and endY == 0), "function unable to detect invalid shape type"
    else:
        assert (no_of_faces == 0), "function unable to detect invalid shape type"

def test_face_detection_draw_bounding_box_with_invalid_coordinates():

    startX = -50
    startY = -50
    endX = -50
    endY = -50
    model = face_detecter()
    status = model.load_model_and_weight()

    frame_data = np.zeros(shape=(50, 50, 3))
    result = model.draw_bounding_box(frame_data, startX, startY, endX, endY)
    assert result is None, "function unable to detect invalid coordinates"