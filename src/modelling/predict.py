import numpy as np
from src.modelling.preprocessing import convert_img_to_grey, resize_image, reshape_image_for_tensor, normalize, drawLabel

class Predictions():
    def __init__(self):
        """
        Parameters used for prediction
        """
        self.dict={'people':['KahGhi','Trump','Obama','YuYang','Simon'],
                    'gender':['Female', 'Male'],
                    'emotion':['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']}
        self.emo_size = (48,48,1)
        self.gender_size = (224,224,3)

    def predict_identity(self, model, frame, startX, startY, endX, endY):
        """
        Takes in model, frame, and the four coordinates of the bounding box
        Distance score is inverse, the more similar the person is, the lower the distance score

        :return: Text drawn on video frame with person's name and distance score
        """
        grey_frame = convert_img_to_grey(frame)
        grey_frame = grey_frame[startY:endY,startX:endX]
        pred,confidence = model.predict(grey_frame)
        if confidence > 80:
            label = "Ambiguous"
        else:
            label = self.dict['people'][pred]
        drawLabel(frame, 
                        "{}, Distance:{:4.2f}".format(label, confidence),
                        x = startX,
                        y = startY-55)

    def predict_gender(self, model, frame, startX, startY, endX, endY):
        """
        Takes in model, frame, and the four coordinates of the bounding box
        
        :return: Text drawn on video frame with predicted gender and confidence score
        """
        grey_frame1 = frame[startY:endY,startX:endX]
        grey_frame1 = resize_image(grey_frame1, self.gender_size)
        grey_frame1 = reshape_image_for_tensor(grey_frame1, self.gender_size)
        grey_frame1 = normalize(grey_frame1)
        label, confidence = self.predict(model, 'gender', grey_frame1)
        drawLabel(frame, 
                        "{}, Score:{:4.2f}".format(label, confidence),
                        x = startX,
                        y = startY-15)

    def predict_emo(self, model, frame, startX, startY, endX, endY):
        """
        Takes in model, frame, and the four coordinates of the bounding box

        :return: Text drawn on video frame with predicted emotion and confidence score
        """
        grey_frame1 = convert_img_to_grey(frame)
        grey_frame1 = grey_frame1[startY:endY,startX:endX]
        grey_frame1 = resize_image(grey_frame1, self.emo_size)
        grey_frame1 = reshape_image_for_tensor(grey_frame1, self.emo_size)
        grey_frame1 = normalize(grey_frame1)
        label, confidence = self.predict(model, 'emotion', grey_frame1)
        drawLabel(frame, 
                        "{}, Score:{:4.2f}".format(label, confidence),
                        x = startX,
                        y = startY - 35)

    def predict(self,model,label,grey_image:np.array):
        """
        Takes in model, category label, and image

        :return: Predicted label, confidence score
        """
        pred = model.predict(grey_image)
        predictions = self.dict[label][np.argmax(pred)]
        score = np.max(pred)
        return predictions , score
