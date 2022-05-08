import numpy as np
import cv2

import logging
logging.basicConfig(level=logging.INFO)

class face_detecter():

    def __init__ (self):
        """
        Instantiation of face detection model

        :return: not applicable
        """
        
        self.prototxt_path = r"./model/face_detection/deploy.prototxt.txt"
        self.model_path = r"./model/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
        self.bb_confidence_threshold = 0.8
        self.model = None

    def load_model_and_weight(self):
        """
        Loading Caffe Model

        :return: not applicable
        """

        logging.info('[Status] Loading Model...')
        try:
            self.model = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
            return "ok"
            
        except Exception as e:
            logging.error("Error in loading model's weight")
            logging.error(str(e))
            return None

    def detect_face(self, frame):
        """
        Take the video frame, try to perform face detection and,
        draw the bounding boxes with confidence level if face was detected

        :return: the four coordinates of the face component
                 startX, startY, endX, endY
        """

        # Initialize variables
        startX = 0
        startY = 0
        endX = 0
        endY = 0
        faces = []

        # Check if frame is num array and shape of 3 (image RGB)
        if (type(frame) != np.ndarray):
            logging.error("Frame is not a numpy array")
            faces.append((startX, startY, endX, endY))
            return faces
        if (len(frame.shape) != 3 ):
            logging.error("Frame is not a RGB image")
            faces.append((startX, startY, endX, endY))
            return faces

        # Get the height and width of frame
        (h, w) = frame.shape[:2]

        # Converting Frame to Blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
            1.0, (300, 300), (104.0, 177.0, 123.0))

        # Passing Blob through network to detect and predict
        self.model.setInput(blob)
        detections = self.model.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):

            # Extracting the confidence of predictions
            confidence = detections[0, 0, i, 2]

            # Filtering out weak predictions
            if confidence > self.bb_confidence_threshold:
                
                # Extracting bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Store different faces' boxes
                faces.append((startX, startY, endX, endY))

        return faces

    def draw_bounding_box(self, frame, startX, startY, endX, endY, confidence=0):
        """
        Draw the bounding boxes and
        confidence level if confidence is not zero (default)

        :return: not applicable
        """

        if (startX < -10 or startY < 0 or endX < 0 or endY < 0):
            logging.error("Negative coordinates in the bounding box")
            return None

        # Draw the bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # Draw the confidence level
        if (confidence != 0):

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y), self.video_font, 0.5, (0, 0, 255), 2)
