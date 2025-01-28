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
        self.landmarks["eyes_landmarks"] = []
        self.landmarks["eyebrows_landmarks"] = []
        self.landmarks["left_iris_landmarks"] = []
        self.landmarks["right_iris_landmarks"] = []
        self.landmarks["nose_landmarks"] = []
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
        self.landmarks["face_points"] = []
        
    def relativeValue(self, data, img, oak=False):
        self.landmarks["face_points"] = []
        pnt_zero = (0,0)
        out_data = []
        for i, lm in enumerate(data):
            h, w, ic = img.shape
            if oak:
                x, y = int(lm[0]), int(lm[1])
            else:
                x, y = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
            if i == 0:
                pnt_zero = (x, y)
            else:
                b = (x, y)
                dst = distance.euclidean(pnt_zero, b)
                out_data.append(dst)
            
            self.landmarks["face_points"].append((x, y))
            
        return out_data
        
    def sortData(self, faceLms, img, oak=False):
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
        
        self.landmarks["eyes_landmarks"] = []
        self.landmarks["eyebrows_landmarks"] = []
    
        self.landmarks["left_eyebrow"] = [] 
        self.landmarks["right_eyebrow"] = [] 
        self.landmarks["mouth_openness"] = []
        self.landmarks["left_eye_openness"] = []
        self.landmarks["right_eye_openness"] = []
        self.landmarks["nose_height"] = []
        self.landmarks["nose_width"] = []
        self.landmarks["all_landmarks"] = [] 
        
        # Iterate over detected faces (here, max_num_faces = 1, so usually one face)
        relative_landmarks = []
        if oak:
            relative_landmarks = self.relativeValue(faceLms, img, oak)
        else:
            relative_landmarks = self.relativeValue(faceLms.landmark, img, oak)
        
        for i, pnt in enumerate(relative_landmarks):
            # Store the coordinates of all landmarks
            self.landmarks["all_landmarks"].append(pnt)

            # Store specific feature landmarks based on the predefined indices
            if i in self.LEFT_EYE_LANDMARKS:
                self.landmarks["left_eye_landmarks"].append(pnt)  # Left eye
                self.landmarks["eyes_landmarks"].append(pnt)  # Both eyes
            if i in self.RIGHT_EYE_LANDMARKS:
                self.landmarks["right_eye_landmarks"].append(pnt)  # Right eye
                self.landmarks["eyes_landmarks"].append(pnt)  # Both eyes
            if i in self.LEFT_IRIS_LANDMARKS:
                self.landmarks["left_iris_landmarks"].append(pnt)  # Left iris
                self.landmarks["eyes_landmarks"].append(pnt)  # Both eyes
            if i in self.RIGHT_IRIS_LANDMARKS:
                self.landmarks["right_iris_landmarks"].append(pnt)  # Right iris
                self.landmarks["eyes_landmarks"].append(pnt)  # Both eyes
            if i in self.NOSE_LANDMARKS:
                self.landmarks["nose_landmarks"].append(pnt)  # Nose
            if i in self.MOUTH_LANDMARKS:
                self.landmarks["mouth_landmarks"].append(pnt)  # Mouth
            if i in self.FACE_OUTLINE:
                self.landmarks["face_outline"].append(pnt)  # Mouth
            if i in self.LEFT_EYEBROW_LANDMARKS:
                self.landmarks["left_eyebrow"].append(pnt)
                self.landmarks["eyebrows_landmarks"].append(pnt)
                
            if i in self.RIGHT_EYEBROW_LANDMARKS:
                self.landmarks["right_eyebrow"].append(pnt)
                self.landmarks["eyebrows_landmarks"].append(pnt)
                
            if i in self.MOUTH_OPENNESS_LANDMARKS:
                self.landmarks["mouth_openness"].append(pnt)
            if i in self.LEFT_EYE_OPENNESS_LANDMARKS:
                self.landmarks["left_eye_openness"].append(pnt)
            if i in self.RIGHT_EYE_OPENNESS_LANDMARKS:
                self.landmarks["right_eye_openness"].append(pnt)
            if i in self.NOSE_HEIGHT_LANDMARKS:
                self.landmarks["nose_height"].append(pnt)
            if i in self.NOSE_WIDTH_LANDMARKS:
                self.landmarks["nose_width"].append(pnt)
                    
        return self.landmarks
        
    def trainProcess(self):
        print("Process based on training data")
        
    
            
    
    