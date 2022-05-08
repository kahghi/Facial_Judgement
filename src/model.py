import tensorflow as tf
import cv2

class Models():
    def __init__(self):
        self.face_path=r'.\model\face_recognition\trained_weights.yml'
        self.gender_path=r'.\model\gender_recognition\gender_mobileNet.hdf5'
        self.emo_path=r'.\model\emo_recognition\emo_VGG19.hdf5'

        self.face_model = self.load_fr_model(self.face_path)
        self.gender_model = self.load_model(self.gender_path)
        self.emo_model = self.load_model(self.emo_path)    

    def load_model(self, path):
        """
        Takes in file path to model
        
        :return: loaded model
        """
        return tf.keras.models.load_model(path)
    
    def load_fr_model(self, path):
        """
        Takes in file path to trained weights
        
        :return: loaded model
        """
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(path)
        return model