import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

from PIL import Image
from keras.preprocessing.image import img_to_array

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = FacialExpressionModel("model_a.json", "model_weights.h5")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def Detect(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=10)
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            edges = cv2.Canny(fc, 110, 1000)
            number_of_edges = np.count_nonzero(edges)

            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            st.write("Predicted Emotion is " + pred)
            if number_of_edges >= 1000:
                st.write("Wrinkle Found ")
                text = pred + " and Wrinkle Found"
            else:
                st.write("No Wrinkle Found ")
                text = pred + " and No Wrinkle Found"
            st.write(text)

    except:
        st.write("Unable to detect, Wrinkle can't be detected!...")

def live():
    cap = cv2.VideoCapture(0)
    mod = load_model("model.h5")
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = mod.predict(roi)[0]
                label = EMOTIONS_LIST[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_image():
    try:
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if file is not None:
            image = Image.open(file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            Detect(image)
    except:
        st.write("File not found")
    
def speech():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write('Clearing background noise...')
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.write('Waiting for your message...')
        recordedaudio = recognizer.listen(source)
        st.write('Done recording..')
    try:
        st.write('Printing the message..')
        text = recognizer.recognize_google(recordedaudio, language='en-US')
        st.write('Your message: {}'.format(text))

        Sentence = [str(text)]
        analyser = SentimentIntensityAnalyzer()
        for i in Sentence:
            v = analyser.polarity_scores(i)
            st.write(v)
            st.write(max(v))
    except Exception as ex:
        st.write("Unable to genereate")

def drowsiness():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    first_read = True
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 1, 1)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_face = gray[y:y+h, x:x+w]
                roi_face_clr = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
    
                if len(eyes) >= 2:
                    if first_read:
                        live()
                    else:
                        cv2.waitKey(3000)
                        first_read = True
                else:
                    if first_read:
                        speech()

        cv2.imshow('img', img)
        a = cv2.waitKey(1)

        if a == ord('q'):
            break
        elif a == ord('s') and first_read:
            first_read = False

    cap.release()
    cv2.destroyAllWindows()

st.title('Emotion Detector')

upload = st.button("Upload Image", on_click=upload_image)
video = st.button("Real time Video - Emotion Detection", on_click=live)
voice = st.button("Voice tone analysis - Emotion Detection", on_click=speech)
drowsy = st.button("Drowsiness Detection", on_click=drowsiness)
