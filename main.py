import contextlib
import depthai as dai
from depthai_sdk import OakCamera, Visualizer, TextPosition
from depthai_sdk.classes.packets  import TwoStagePacket
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
import threading
import time

osc_port = 8000

face_active = False
face_out_time = 0

def run_camera(device_id, emotion_data):
    global face_active
    with OakCamera(device_id) as oak:
        smooth_emotion = [0,0,0,0,0]
        
        color = oak.create_camera('color')
        det_nn = oak.create_nn('models/face-detection-retail-0004_openvino_2022.1_6shave.blob', color, nn_type="mobilenet")
        det_nn.config_nn(resize_mode='crop')

        emotion_nn = oak.create_nn('models/emotions-recognition-retail-0003_openvino_2022.1_6shave.blob', input=det_nn)

        def cb(packet: TwoStagePacket):
            for det, rec in zip(packet.detections, packet.nnData):
                emotion_results = np.array(rec.getFirstLayerFp16())
                #Smooths the emotion results
                for i in range(len(emotion_results)):
                    smooth_emotion[i] = 0.9 * smooth_emotion[i] + 0.1 * emotion_results[i]
                    emotion_data[device_id][i] = smooth_emotion[i]
                    
                frame_center = packet.frame.shape[0] / 2
                head_y = int((det.top_left[1] + det.bottom_right[1]) / 2)
                head_x = int((det.top_left[0] + det.bottom_right[0]) / 2)
                
                face_dist = abs(frame_center - head_y) / frame_center
                
                emotion_data[device_id][5] = face_dist
                emotion_data[device_id][6] = time.time()
                emotion_data[device_id][7] = head_x / packet.frame.shape[1]
                if emotion_data[device_id][7] > 1:
                    emotion_data[device_id][7] = 1
                    
                if emotion_data[device_id][7] < 0:
                    emotion_data[device_id][7] = 0
                
                            
        oak.visualize(emotion_nn, callback=cb, fps=True)    
        #oak.visualize(det_nn.out.passthrough)
        oak.start(blocking=True)

threads = []
emotion_data = {}
client = SimpleUDPClient("127.0.0.1", osc_port)

def send_osc(emotions):
    global face_active, face_out_time
    if not face_active:
        client.send_message("/active", 1)    
    
    client.send_message("/emotion/happy", float(emotions[1]))
    client.send_message("/emotion/angry", float(emotions[4]))
    client.send_message("/emotion/sad", float(emotions[2]))
    client.send_message("/emotion/suprised", float(emotions[3]))
    client.send_message("/emotion/neutral", float(emotions[0]))
    sadfix = float(emotions[2]) - float(emotions[0])
    if sadfix < 0:
        sadfix = 0
    client.send_message("/emotion/sadfix", sadfix)
    client.send_message("/xpos", abs(emotions[7] - 1))
    face_out_time = 0
    face_active = True

with contextlib.ExitStack() as stack:
    
    device_infos = dai.Device.getAllAvailableDevices()
    
    while len(device_infos) != 1:
        print("Only found", len(device_infos), "device and will check again")
        time.sleep(4)
        device_infos = dai.Device.getAllAvailableDevices()

    print("Found", len(device_infos), "devices")
        
    for device_info in device_infos:
        emotion_data[device_info.getMxId()] = [0,0,0,0,0,0,0,0]
        
    for device_info in device_infos:
        print("Connecting to device id: ", device_info.getMxId())
        t = threading.Thread(target=run_camera, args=(device_info.getMxId(), emotion_data,))
        t.start()
        threads.append(t)
        
    while True:
        detected_faces = []
        for device in device_infos:
            #Check if we have seen a face in the last 0.5 seconds
            if time.time() - emotion_data[device.getMxId()][6] < 0.5:
                detected_faces.append(device.getMxId())
        
       
        if len(detected_faces) > 1:
            # I f we see multiple faces, send the one closest to the center
            min_face_dist = float('inf')
            selected_device = None
            for device in detected_faces:
                if emotion_data[device][5] < min_face_dist:
                    min_face_dist = emotion_data[device][5]
                    selected_device = device
            if selected_device:
                send_osc(emotion_data[selected_device])
        else:
            # If we only see one face send that
            if detected_faces:
                send_osc(emotion_data[detected_faces[0]])
            else:
                if face_active:
                    face_out_time += 1
                    if face_out_time > 50:
                        client.send_message("/active", 0)
                        face_active = False
           
        time.sleep(0.01)
        
for t in threads:
    t.join()
