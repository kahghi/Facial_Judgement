
import numpy as np
from PIL import Image
from tensorflow import keras
# ------------------------------------------------------------------
# Remarks: take note that running pytest from assignment8 folder
#          command: pytest ./tests
# ------------------------------------------------------------------
# Constant variables used in these test
emotion =  ['Male','Female']
model_base_folder = './model/gender_recognition/'
images_base_folder = './images/'
model_filename = 'gender_mobileNet.hdf5'

# ===========
# Positive TC
# ===========
def test_gender_model_predict_female():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "1_1_0_20170109190742179.jpg"
    img = Image.open(image_filename)
    img = img.resize((224, 224))

    inp = np.reshape(img,(1,224,224,3)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Female' in predict_class, 'predicted class is not [Female]'

def test_gender_model_predict_male():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "16_0_0_20170110232142982.jpg"
    img = Image.open(image_filename)
    img = img.resize((224, 224))

    inp = np.reshape(img,(1,224,224,3)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Male' in predict_class, 'predicted class is not [Male]'