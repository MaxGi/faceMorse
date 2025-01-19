import cv2
import mediapipe as mp
from process_landmarks import FaceAalysis

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
face_analysis = FaceAalysis()

# Open webcam
cap = cv2.VideoCapture(1)

training = True

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find face mesh
    results = face_mesh.process(image_rgb)

    # Draw the face mesh annotations on the image
    if results.multi_face_landmarks:
        
        for face_landmarks in results.multi_face_landmarks: 
            face_data = face_analysis.sortData(face_landmarks, image)           
            for pt in face_data["all_landmarks"]:
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
        
        text_in = input("Classify: ")
        if text_in != "":
            print("Got input: ", text_in)
            
            
    # Display the image
    cv2.imshow('Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()