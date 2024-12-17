from depthai_sdk import OakCamera, TextPosition, Visualizer
from depthai_sdk.classes.packets  import TwoStagePacket
import numpy as np
import cv2
from pythonosc.udp_client import SimpleUDPClient

show_visuals = True

#Change to correct device ID found in find_ids.py
device_id = "1844301081D4420E00"
osc_port = 8000

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
smooth_emotion = [0,0,0,0,0]

client = SimpleUDPClient("127.0.0.1", osc_port)

#Opens camera based on the device ID
with OakCamera(device_id) as oak:
    
    
    color = oak.create_camera('color')
    color.setBrightness(5)
    det_nn = oak.create_nn('models/face-detection-retail-0004_openvino_2022.1_6shave.blob', color, nn_type="mobilenet")
    det_nn.config_nn(resize_mode='crop')

    emotion_nn = oak.create_nn('models/emotions-recognition-retail-0003_openvino_2022.1_6shave.blob', input=det_nn)

    def cb(packet: TwoStagePacket):
        vis: Visualizer = packet.visualizer
        
        _dist = []
        
        for det, rec in zip(packet.detections, packet.nnData):
            emotion_results = np.array(rec.getFirstLayerFp16())
            for i in range(len(emotion_results)):
                smooth_emotion[i] = 0.9 * smooth_emotion[i] + 0.1 * emotion_results[i]
            
            client.send_message("/emotion/happy", float(smooth_emotion[1]))
            client.send_message("/emotion/angry", float(smooth_emotion[4]))
            client.send_message("/emotion/sad", float(smooth_emotion[2]))
            client.send_message("/emotion/suprised", float(smooth_emotion[3]))
            client.send_message("/emotion/neutral", float(smooth_emotion[0]))
            
            sadfix = float(smooth_emotion[2]) - float(smooth_emotion[0])
            if sadfix < 0:
                sadfix = 0
            
            client.send_message("/emotion/sadfix", sadfix)
            
            frame_center = packet.frame.shape[0] / 2
            head_y = int((det.top_left[1] + det.bottom_right[1]) / 2)
            
            face_dist = abs(frame_center - head_y) / frame_center
            
            _dist.append(face_dist)
            
            emotion_name = emotions[np.argmax(emotion_results)]
            
            if show_visuals:
                vis.add_text(emotion_name,
                                bbox=(*det.top_left, *det.bottom_right),
                                position=TextPosition.BOTTOM_RIGHT)
        if show_visuals:
            vis.draw(packet.frame)
            for center in _dist:
                cv2.putText(packet.frame, str(round(center, 4)), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            cv2.imshow("image", packet.frame)
            
    # Visualize detections on the frame. Also display FPS on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(emotion_nn, callback=cb, fps=True)    
    oak.visualize(det_nn.out.passthrough)
    oak.start(blocking=True) # This call will block until the app is stopped (by pressing 'Q' button)