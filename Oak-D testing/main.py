import cv2
import depthai as dai
import math
from typing import List
from ultralytics import YOLO  # Your v11 model here

# ---- Constants ----
LENS_STEP = 3
DEBUG = True
MODEL_PATH = "/home/abhinav/fuel/yolo_trained_weights/best_v11.pt"  # Replace this with actual path

# ---- Utility Classes ----

class TextHelper:
    def __init__(self):
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.5, self.bg_color, 6, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.5, self.color, 2, self.line_type)

    def rectangle(self, frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 6)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)

class HostSync:
    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if name not in self.arrays:
            self.arrays[name] = []
        self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

        synced = {}
        for name, arr in self.arrays.items():
            for obj in arr:
                if msg.getSequenceNum() == obj["seq"]:
                    synced[name] = obj["msg"]
                    break

        if len(synced) == (2 if DEBUG else 1):
            for name in self.arrays:
                self.arrays[name] = [obj for obj in self.arrays[name] if obj["seq"] >= msg.getSequenceNum()]
            return synced
        return False

# ---- Functions ----

def calculate_distance(coords):
    return math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

def get_lens_position(dist):
    return int(150 - dist * 0.0242 + 0.00000412 * dist**2)

def create_pipeline():
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName('control')
    controlIn.out.link(cam_rgb.inputControl)

    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_video.setStreamName("color")
    cam_rgb.video.link(xout_video.input)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setConfidenceThreshold(240)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setExtendedDisparity(True)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

# ---- Main Execution ----

model = YOLO(MODEL_PATH)

with dai.Device(create_pipeline()) as device:
    controlQ = device.getInputQueue('control')
    queues = {name: device.getOutputQueue(name, 4, False) for name in ['color', 'depth']}
    sync = HostSync()
    text = TextHelper()
    lensPos = 150
    lensMin, lensMax = 0, 255

    while True:
        for name, q in queues.items():
            if q.has():
                synced_msgs = sync.add_msg(name, q.get())
                if synced_msgs:
                    frame = synced_msgs['color'].getCvFrame()
                    depth_frame = synced_msgs['depth'].getFrame()

                    # Run YOLOv11 on RGB frame
                    results = model(frame)[0]
                    detections = results.boxes.xyxy.cpu().numpy().astype(int)

                    depth_vis = cv2.pyrDown(depth_frame)
                    depth_vis = cv2.normalize(depth_vis, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depth_vis = cv2.equalizeHist(depth_vis)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)

                    closest_dist = float('inf')

                    for det in detections:
                        x1, y1, x2, y2 = det
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        if 0 <= cy < depth_frame.shape[0] and 0 <= cx < depth_frame.shape[1]:
                            depth_pixel = depth_frame[cy][cx]
                            coords = dai.Point3f(0, 0, depth_pixel)
                            dist = calculate_distance(coords)

                            if dist < closest_dist:
                                closest_dist = dist=[]

                            text.rectangle(frame, x1, y1, x2, y2)
                            text.rectangle(depth_vis, x1, y1, x2, y2)

                    if closest_dist != float('inf'):
                        text.putText(frame, f"Distance: {closest_dist/1000:.2f} m", (30, 1045))
                        new_lens_pos = clamp(get_lens_position(closest_dist), lensMin, lensMax)
                        if new_lens_pos != lensPos and new_lens_pos != 255:
                            lensPos = new_lens_pos
                            ctrl = dai.CameraControl()
                            ctrl.setManualFocus(lensPos)
                            controlQ.send(ctrl)
                    else:
                        text.putText(frame, "Distance: /", (30, 1045))

                    text.putText(frame, f"Lens position: {lensPos}", (30, 1000))

                    cv2.imshow("RGB", cv2.resize(frame, (750, 750)))
                    cv2.imshow("Depth", depth_vis)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key in [ord(','), ord('.')]:
            lensPos += LENS_STEP if key == ord('.') else -LENS_STEP
            lensPos = clamp(lensPos, lensMin, lensMax)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            controlQ.send(ctrl)
