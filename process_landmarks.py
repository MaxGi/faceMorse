from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

class FaceAalysis:
    def __init__(self):
        
        self.LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362]  # Left eye landmarks

        self.RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7]  # Right eye landmarks

        self.LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]  # Left iris landmarks
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Right iris landmarks

        self.NOSE_LANDMARKS = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48,
                               278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274]  # Nose landmarks

        self.MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39,
                                37]
        
        self.FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        self.LEFT_EYEBROW_LANDMARKS = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]  # Left eyebrow landmarks
        self.RIGHT_EYEBROW_LANDMARKS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]  # Right eyebrow landmarks
        
        self.MOUTH_OPENNESS_LANDMARKS = [13, 14]  # Top lip and bottom lip landmarks
        self.LEFT_EYE_OPENNESS_LANDMARKS = [386, 374]  # Top and bottom landmarks for left eye openness
        self.RIGHT_EYE_OPENNESS_LANDMARKS = [159, 145]  # Top and bottom landmarks for right eye openness
        self.NOSE_HEIGHT_LANDMARKS = [1, 2]  # Top and bottom landmarks for nose height
        
        self.NOSE_WIDTH_LANDMARKS = [49, 279]  # Left and right landmarks for nose width
        
        self.landmarks = {}
        self.landmarks["left_eye_landmarks"] = []
        self.landmarks["right_eye_landmarks"] = []
        self.landmarks["left_iris_landmarks"] = []
        self.landmarks["right_iris_landmarks"] = []
        self.landmarks["nose_landmarks"] = []
        self.landmarks["mouth_landmarks"] = []
        self.landmarks["mouth_landmarks"] = []
        self.landmarks["head_border_landmarks"] = []
        self.landmarks["face_outline"] = [] 
    
        self.landmarks["left_eyebrow"] = [] 
        self.landmarks["right_eyebrow"] = [] 
        self.landmarks["mouth_openness"] = []
        self.landmarks["left_eye_openness"] = []
        self.landmarks["right_eye_openness"] = []
        self.landmarks["nose_height"] = []
        self.landmarks["nose_width"] = []
        self.landmarks["all_landmarks"] = [] 
        
        
        
    def sortData(self, faceLms, img):
        self.landmarks = {}
        self.landmarks["left_eye_landmarks"] = []
        self.landmarks["right_eye_landmarks"] = []
        self.landmarks["left_iris_landmarks"] = []
        self.landmarks["right_iris_landmarks"] = []
        self.landmarks["nose_landmarks"] = []
        self.landmarks["mouth_landmarks"] = []
        self.landmarks["mouth_landmarks"] = []
        self.landmarks["head_border_landmarks"] = []
        self.landmarks["face_outline"] = [] 
    
        self.landmarks["left_eyebrow"] = [] 
        self.landmarks["right_eyebrow"] = [] 
        self.landmarks["mouth_openness"] = []
        self.landmarks["left_eye_openness"] = []
        self.landmarks["right_eye_openness"] = []
        self.landmarks["nose_height"] = []
        self.landmarks["nose_width"] = []
        self.landmarks["all_landmarks"] = [] 
        # Iterate over detected faces (here, max_num_faces = 1, so usually one face)
        
        for i, lm in enumerate(faceLms.landmark):
            
            h, w, ic = img.shape  # Get image height, width, and channel count
            x, y = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
            
            # Store the coordinates of all landmarks
            self.landmarks["all_landmarks"].append((x, y))

            # Store specific feature landmarks based on the predefined indices
            if i in self.LEFT_EYE_LANDMARKS:
                self.landmarks["left_eye_landmarks"].append((x, y))  # Left eye
            if i in self.RIGHT_EYE_LANDMARKS:
                self.landmarks["right_eye_landmarks"].append((x, y))  # Right eye
            if i in self.LEFT_IRIS_LANDMARKS:
                self.landmarks["left_iris_landmarks"].append((x, y))  # Left iris
            if i in self.RIGHT_IRIS_LANDMARKS:
                self.landmarks["right_iris_landmarks"].append((x, y))  # Right iris
            if i in self.NOSE_LANDMARKS:
                self.landmarks["nose_landmarks"].append((x, y))  # Nose
            if i in self.MOUTH_LANDMARKS:
                self.landmarks["mouth_landmarks"].append((x, y))  # Mouth
            if i in self.FACE_OUTLINE:
                self.landmarks["face_outline"].append((x, y))  # Mouth
            if i in self.LEFT_EYEBROW_LANDMARKS:
                self.landmarks["left_eyebrow"].append((x, y))
            if i in self.RIGHT_EYEBROW_LANDMARKS:
                self.landmarks["right_eyebrow"].append((x, y))
            if i in self.MOUTH_OPENNESS_LANDMARKS:
                self.landmarks["mouth_openness"].append((x, y))
            if i in self.LEFT_EYE_OPENNESS_LANDMARKS:
                self.landmarks["left_eye_openness"].append((x, y))
            if i in self.RIGHT_EYE_OPENNESS_LANDMARKS:
                self.landmarks["right_eye_openness"].append((x, y))
            if i in self.NOSE_HEIGHT_LANDMARKS:
                self.landmarks["nose_height"].append((x, y))
            if i in self.NOSE_WIDTH_LANDMARKS:
                self.landmarks["nose_width"].append((x, y))
                    
        return self.landmarks
        
    def trainProcess(self):
        print("Process based on training data")
        
    
            
    
    