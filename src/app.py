#https://github.com/amolikvivian/Caffe-SSD-Object-Detection/blob/master/Object%20Detection%20Caffe/detectDNN.py
#https://www.youtube.com/watch?v=piaEXzNkowY
import time

import cv2
import imutils
from imutils.video import FPS
from imutils.video import VideoStream
from src.modelling.face_detection.face_detect import face_detecter
from src.model import Models
from src.modelling.predict import Predictions 

import logging
logging.basicConfig(level=logging.INFO)

class Inference():

    def __init__(self):
        """
        Instantiation of model & video related parameters

        :return: not applicable
        """
        # # Init face detection
        self.fd_model = face_detecter()
        self.fd_model.load_model_and_weight()
        # Init models
        self.fr_model = Models().face_model
        self.er_model = Models().emo_model
        self.gd_model = Models().gender_model
        
        # Video related parameters
        self.video_width = 800
        self.video_font = cv2.FONT_HERSHEY_SIMPLEX
        self.video_terminate_flag = False

    def start_video_stream(self, flag=False):
        """
        This function is used for standalone application
        Start video stream and perform the following
        1. face detection
        2. facial recognition
        3. emotion recognition
        4. gender recognition 

        :return: not applicable
        """

        # Initialize Video Stream
        logging.info('[Status] Starting Video Stream...')
        cam = VideoStream(src=0).start()
        time.sleep(2.0)
        fps = FPS().start()                                                 # For computing frame per second (not necessary) 


        # Loop Video Stream
        while True:

            frame = cam.read()
            frame = cv2.flip(frame,1)
            frame = imutils.resize(frame, width = self.video_width)         # Resize frame to video width
            
            faces = self.fd_model.detect_face(frame)
            for (startX, startY, endX, endY) in faces:

                self.fd_model.draw_bounding_box(frame, startX, startY, endX, endY)
                Predictions().predict_identity(self.fr_model, frame, startX, startY, endX, endY)
                Predictions().predict_emo(self.er_model, frame, startX, startY, endX, endY)
                Predictions().predict_gender(self.gd_model, frame, startX, startY, endX, endY)

            # Show video frame
            cv2.imshow("Frame", frame)

            # Wait for keyboard 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            fps.update()                                                    # For computing frame per second (not necessary) 

        fps.stop()                                                          # For computing frame per second (not necessary) 

        logging.info("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))   # For computing frame per second (not necessary) 
        logging.info("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))   # For computing frame per second (not necessary) 

        cv2.destroyAllWindows()
        cam.stream.release()
        logging.info("[Info] Camera resourse released")

    def start_video_stream_web(self):
        """
        This function is used for flask web application
        Start video stream and perform the following
        1. face detection
        2. facial recognition
        3. emotion recognition
        4. gender recognition 

        :return: not applicable
        """

        # Initialize Video Stream
        logging.info('[Status] Starting Video Stream...')
        self.video_terminate_flag = False
        cam = VideoStream(src=0).start()
        time.sleep(2.0)

        # Loop Video Stream
        while True:

            frame = cam.read()
            frame = imutils.resize(frame, width = self.video_width)         # Resize frame to video width
            
            faces = self.fd_model.detect_face(frame)
            for (startX, startY, endX, endY) in faces:

                self.fd_model.draw_bounding_box(frame, startX, startY, endX, endY)
                Predictions().predict_identity(self.fr_model, frame, startX, startY, endX, endY)
                Predictions().predict_emo(self.er_model, frame, startX, startY, endX, endY)
                Predictions().predict_gender(self.gd_model, frame, startX, startY, endX, endY)

            # Convert frame to image bytes
            frame_byte = self.convert_frame_to_byte(frame)

            # Return jpg image byte to web client
            yield(b'--frame\r\n'
                    b'Content-Type:image/jpeg\r\n\r\n' + frame_byte + b'\r\n')   
            
            # Wait for terminate flag to be set
            if (self.video_terminate_flag == True):
                break

        cv2.destroyAllWindows()
        cam.stream.release()
        logging.info("[Info] Camera resourse released")

    def video_terminate(self):
        """
        This function is used for flask web application
        Set the terminate flag when user click on "stop stream"

        :return: not applicable
        """

        logging.info("[Status] Terminating..")
        self.video_terminate_flag = True

        return False

    def convert_frame_to_byte(self, frame):
        """
        This function is used for flask web application
        Convert the video frame to image byte to send back web browser

        :return: not applicable
        """

        # Convert frame to jpg image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_byte = buffer.tobytes()

        return frame_byte

if __name__ == '__main__':
    
    inf = Inference()
    inf.start_video_stream()
