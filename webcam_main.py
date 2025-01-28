import cv2
import mediapipe as mp
from process_landmarks import FaceAalysis
from model import Model

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
face_analysis = FaceAalysis()

# Open webcam
cap = cv2.VideoCapture(0)

mouth_model = Model(model_name="mouth_model", data_key="mouth_landmarks", model_type="regression")
nose_model = Model(model_name="nose_model", data_key="nose_landmarks", model_type="regression")
eyes_model = Model(model_name="eye_model", data_key="eyes_landmarks")
eybrow_model = Model(model_name="eybrow_model", data_key="eyebrows_landmarks")
mouth_angle_model = Model(model_name="mouth_angle_model", data_key="mouth_landmarks")


training = False

if not training:
    mouth_model.loadModel()
    nose_model.loadModel()
    eyes_model.loadModel()
    eybrow_model.loadModel()
    mouth_angle_model.loadModel()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find face mesh
    results = face_mesh.process(image_rgb)

    
    face_data = None
    
    # Draw the face mesh annotations on the image
    if results.multi_face_landmarks:
        
        for face_landmarks in results.multi_face_landmarks: 
            face_data = face_analysis.sortData(face_landmarks, image)           
            print("DATA: ", face_landmarks)
            for pt in face_data["face_points"]:
                x = pt[0]
                y = pt[1]
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    
    if training:
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)
        cv2.putText(image, 'Training Mode', (20, 50), font, 1, color, 1, cv2.LINE_AA)
        cv2.putText(image, 'Training Mode', (20, 80), font, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(image, 'Training Mode', (20, 100), font, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(image, 'Training Mode', (20, 120), font, 0.5, color, 1, cv2.LINE_AA)
            
    # Display the image
    cv2.imshow('Face Mesh', image)
    
    if face_data is not None and not training:
        prediction = mouth_model.predict(face_data)
        print("Prediction Mouth:", prediction)
        prediction = eyes_model.predict(face_data)
        print("Prediction Eyes:", prediction)
        prediction = nose_model.predict(face_data)
        print("Prediction Nose:", prediction)
        prediction = eybrow_model.predict(face_data)
        print("Prediction Eyebrow:", prediction)
        prediction = mouth_angle_model.predict(face_data)
        print("Prediction Outline:", prediction)
        
    
    if training:
        text_in = input("Classify: ")
        if text_in != "":
            text_in_split = text_in.split()
            # GO SMALL AND LARGE
            if text_in_split[0] == "0":
                print("Add mouth")
                mouth_model.rec(face_data, text_in_split[1])
            if text_in_split[0] == "1":
                print("Add eye")
                eyes_model.rec(face_data, text_in_split[1])
            if text_in_split[0] == "2":
                print("Add nose")
                nose_model.rec(face_data, text_in_split[1])
            if text_in_split[0] == "3":
                print("Add eybrow")
                eybrow_model.rec(face_data, text_in_split[1])
            if text_in_split[0] == "4":
                print("Add eybrow")
                mouth_angle_model.rec(face_data, text_in_split[1])
            elif text_in_split[0] == "s":
                print("Train")
                mouth_model.train()
                eyes_model.train()
                nose_model.train()
                eybrow_model.train()
                mouth_angle_model.train()
                training = False
                mouth_model.saveModel()
                eyes_model.saveModel()
                nose_model.saveModel()
                eybrow_model.saveModel()
                mouth_angle_model.saveModel()
                
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()