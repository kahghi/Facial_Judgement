
import numpy as np
from PIL import Image
from tensorflow import keras
# ------------------------------------------------------------------
# Remarks: take note that running pytest from assignment8 folder
#          command: pytest ./tests
# ------------------------------------------------------------------
# Constant variables used in these test
emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model_base_folder = './model/emo_recognition/'
images_base_folder = './images/'
model_filename = 'emo_VGG19.hdf5'

# ===========
# Positive TC
# ===========
def test_emotion_model_predict_anger():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "test1.jpg"
    img = Image.open(image_filename).convert('L')
    img = img.resize((48, 48))

    inp = np.reshape(img,(1,48,48,1)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Anger' in predict_class, 'predicted class is not [Anger]'

def test_emotion_model_predict_disgust():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "test2.png"
    img = Image.open(image_filename).convert('L')
    img = img.resize((48, 48))

    inp = np.reshape(img,(1,48,48,1)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Disgust' in predict_class, 'predicted class is not [Disgust]'

def test_emotion_model_predict_fear():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "test3.jpg"
    img = Image.open(image_filename).convert('L')
    img = img.resize((48, 48))

    inp = np.reshape(img,(1,48,48,1)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Fear' in predict_class, 'predicted class is not [Fear]'

def test_emotion_model_predict_happy():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "test4.jpg"
    img = Image.open(image_filename).convert('L')
    img = img.resize((48, 48))

    inp = np.reshape(img,(1,48,48,1)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Happy' in predict_class, 'predicted class is not [Happy]'

def test_emotion_model_predict_sad():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "test5.jpg"
    img = Image.open(image_filename).convert('L')
    img = img.resize((48, 48))

    inp = np.reshape(img,(1,48,48,1)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Sad' in predict_class, 'predicted class is not [Sad]'

def test_emotion_model_predict_surprise():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "test6.jpg"
    img = Image.open(image_filename).convert('L')
    img = img.resize((48, 48))

    inp = np.reshape(img,(1,48,48,1)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Surprise' in predict_class, 'predicted class is not [Surprise]'

def test_emotion_model_predict_netural():

    model_path = model_base_folder + model_filename
    model = keras.models.load_model(model_path)

    image_filename = images_base_folder + "test7.jpg"
    img = Image.open(image_filename).convert('L')
    img = img.resize((48, 48))

    inp = np.reshape(img,(1,48,48,1)).astype(np.float32)
    inp = inp/255.

    prediction = model.predict(inp)
    predict_class = emotion[np.argmax(prediction)]
    score = np.max(prediction)

    assert 'Neutral' in predict_class, 'predicted class is not [Neutral]'