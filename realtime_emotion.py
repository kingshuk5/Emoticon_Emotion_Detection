import cv2
import numpy as np
import joblib
import requests
import os
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

CNN_WEIGHTS_PATH = 'emotion_cnn.weights.h5'
CLASSIFIER_PATH = 'adaboost_emotion_classifier.pkl' 
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
HAAR_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'


def download_haar_cascade():
    """Downloads the Haar Cascade XML file if it doesn't exist."""
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Downloading {HAAR_CASCADE_PATH}...")
        try:
            response = requests.get(HAAR_CASCADE_URL)
            response.raise_for_status()  
            with open(HAAR_CASCADE_PATH, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading Haar Cascade file: {e}")
            return None
    return cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def build_feature_extractor():
    """Builds the CNN model architecture and creates the feature extractor."""
    cnn_model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu', name='feature_layer'),
        Dropout(0.5),
        Dense(8, activation='softmax') 
    ], name="CNN_Model")

    try:
        cnn_model.load_weights(CNN_WEIGHTS_PATH)
        print("CNN weights loaded successfully.")
    except Exception as e:
        print(f"Error loading CNN weights: {e}")
        return None

    
    feature_extractor = Model(
        inputs=cnn_model.layers[0].input,
        outputs=cnn_model.get_layer('feature_layer').output
    )
    return feature_extractor

def load_classifier():
    """Loads the saved scikit-learn classifier."""
    try:
        classifier = joblib.load(CLASSIFIER_PATH)
        print("Classifier loaded successfully.")
        return classifier
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return None


def main():
    face_cascade = download_haar_cascade()
    feature_extractor = build_feature_extractor()
    classifier = load_classifier()

    if face_cascade is None or feature_extractor is None or classifier is None:
        print("Could not initialize all components. Exiting.")
        return

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting webcam feed. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 150, 0), 2)
            roi_gray = gray_frame[y:y+h, x:x+w]

            resized_face = cv2.resize(roi_gray, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
            image_features = feature_extractor.predict(reshaped_face, verbose=0)
            prediction = classifier.predict(image_features)
            predicted_label = EMOTION_LABELS.get(prediction[0], "Unknown")
            cv2.putText(
                frame,
                predicted_label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == '__main__':
    main()
