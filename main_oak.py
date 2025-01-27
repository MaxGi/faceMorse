from depthai_sdk import OakCamera, Visualizer
from depthai_sdk.classes.packets import TwoStagePacket
from depthai_sdk.classes.nn_results import ImgLandmarks
from depthai_sdk.visualize.bbox import BoundingBox
import cv2
import depthai as dai
import screeninfo
import numpy as np

from process_landmarks import FaceAalysis
from model import Model

#import PySimpleGUI as sg


# Confidence threshold for the facemesh model
THRESHOLD = 0.3
# 5% padding for face detection, as facemesh_192x192 is trained on the whole head not just the face
PADDING = 5

# Position of the landmarks in the facemesh model
EyeLeft = 468
EyeRight = 473
NoseTip = 1
NostrilLeft = 205
NostrilRight = 425

screen_id = 0
is_color = False


print(screeninfo.get_monitors())
# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height


smooth = np.full((468, 2), 0)

window_name = 'projector'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

face_analysis = FaceAalysis()

last_landmarks = []

# We will be saving the passthrough frames so we can draw landmarks on it
pass_f = None
def pass_cb(packet):
    global pass_f
    pass_f = packet.frame

def draw_rect(frame, color, top_left, bottom_right):
    cv2.rectangle(frame, top_left, bottom_right, color, 1)

def cb(packet: TwoStagePacket):
    global pass_f
    vis: Visualizer = packet.visualizer
    frame_full = packet.frame

    pre_det_crop_bb = BoundingBox().resize_to_aspect_ratio(frame_full.shape, (1, 1), resize_mode='crop')

    out_frame = np.zeros([720,1280,3],dtype=np.uint8)
    out_frame.fill(0) # or img[:] = 255

    for det, imgLdms in zip(packet.detections, packet.nnData):
        if imgLdms is None or imgLdms.landmarks is None:
            continue
        imgLdms: ImgLandmarks

        img_det: dai.ImgDetection = det.img_detection
        det_bb = pre_det_crop_bb.get_relative_bbox(BoundingBox(img_det))
        
        print("Landmarks: ", imgLdms.landmarks)
        padding_bb = det_bb.add_padding(0.05, pre_det_crop_bb)
        draw_rect(frame_full, (0, 0, 255), *padding_bb.denormalize(frame_full.shape))
        #for i, (name, age) in enumerate(zip(names, ages)):
#        print("list size:", len(imgLdms.landmarks))
        for i, (ldm, clr) in enumerate(zip(imgLdms.landmarks, imgLdms.colors)):
            mapped_ldm = padding_bb.map_point(*ldm).denormalize(frame_full.shape)
            smooth[i][0] = 0.9 * smooth[i][0] + 0.1 * mapped_ldm[0]
            smooth[i][1] = 0.9 * smooth[i][1] + 0.1 * mapped_ldm[1]

            #print(mapped_ldm)
            cv2.circle(out_frame, center=mapped_ldm, radius=2, color=(250, 250, 250), thickness=-1)
            
        face_data = face_analysis.sortData(smooth, out_frame, oak=True) 
        print("DATA: ", face_data)
            
        #if face_data is not None and not training:
            #prediction = eye_model.predict(face_data)
            #print("Prediction", prediction)

    cv2.imshow(window_name, out_frame)

with OakCamera() as oak:

    color = oak.create_camera('color')

    det_nn = oak.create_nn('models/face-detection-retail-0004_openvino_2022.1_6shave.blob', color, nn_type="mobilenet")
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det_nn.config_nn(resize_mode='crop')

#    facemesh_nn = oak.create_nn("models/facemesh_192x192_openvino_2022.1_6shave.blob", input=det_nn, nn_type="mobilenet")
    facemesh_nn = oak.create_nn('facemesh_192x192', input=det_nn)
    facemesh_nn.config_multistage_nn(scale_bb=(5,5))

    oak.visualize(facemesh_nn, callback=cb).detections(fill_transparency=0)

    oak.callback(facemesh_nn.out.twostage_crops, pass_cb)

    oak.start(blocking=True)  # This call will block until the app is stopped (by pressing 'Q' button)
