# from datetime import timedelta
# import blobconverter
# import cv2
# import depthai as dai
# import math
# from typing import List

# LENS_STEP = 3
# DEBUG = True

# class TextHelper:
#     def __init__(self) -> None:
#         self.bg_color = (0, 0, 0)
#         self.color = (255, 255, 255)
#         self.text_type = cv2.FONT_HERSHEY_SIMPLEX
#         self.line_type = cv2.LINE_AA
#     def putText(self, frame, text, coords):
#         cv2.putText(frame, text, coords, self.text_type, 1.5, self.bg_color, 6, self.line_type)
#         cv2.putText(frame, text, coords, self.text_type, 1.5, self.color, 2, self.line_type)
#     def rectangle(self, frame, x1,y1,x2,y2):
#         cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 6)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)


# class HostSync:
#     def __init__(self):
#         self.arrays = {}

#     def add_msg(self, name, msg):
#         if not name in self.arrays:
#             self.arrays[name] = []
#         # Add msg to array
#         self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

#         synced = {}
#         for name, arr in self.arrays.items():
#             for i, obj in enumerate(arr):
#                 if msg.getSequenceNum() == obj["seq"]:
#                     synced[name] = obj["msg"]
#                     break
#         # If there are 5 (all) synced msgs, remove all old msgs
#         # and return synced msgs
#         if len(synced) == (3 if DEBUG else 2):  # Color, Spatial NN results, potentially Depth
#             # Remove old msgs
#             for name, arr in self.arrays.items():
#                 for i, obj in enumerate(arr):
#                     if obj["seq"] < msg.getSequenceNum():
#                         arr.remove(obj)
#                     else:
#                         break
#             return synced
#         return False

# def create_pipeline():
#     print("Creating pipeline...")
#     pipeline = dai.Pipeline()

#     # ColorCamera
#     print("Creating Color Camera...")
#     cam = pipeline.create(dai.node.ColorCamera)
#     cam.setPreviewSize(300, 300)
#     cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#     cam.setVideoSize(1080,1080)
#     cam.setInterleaved(False)

#     controlIn = pipeline.create(dai.node.XLinkIn)
#     controlIn.setStreamName('control')
#     controlIn.out.link(cam.inputControl)

#     left = pipeline.create(dai.node.MonoCamera)
#     left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
#     left.setBoardSocket(dai.CameraBoardSocket.LEFT)

#     right = pipeline.create(dai.node.MonoCamera)
#     right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
#     right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

#     stereo = pipeline.create(dai.node.StereoDepth)
#     stereo.initialConfig.setConfidenceThreshold(240)
#     stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
#     stereo.setExtendedDisparity(True)
#     left.out.link(stereo.left)
#     right.out.link(stereo.right)

#     cam_xout = pipeline.create(dai.node.XLinkOut)
#     cam_xout.setStreamName("color")
#     cam.video.link(cam_xout.input)

#     # NeuralNetwork
#     print("Creating Face Detection Neural Network...")
#     face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
#     face_det_nn.setConfidenceThreshold(0.4)
#     face_det_nn.setBlobPath(blobconverter.from_zoo(
#         name="face-detection-retail-0004",
#         shaves=6,
#         version='2021.4'
#     ))

#     face_det_nn.setBoundingBoxScaleFactor(0.5)
#     face_det_nn.setDepthLowerThreshold(200)
#     face_det_nn.setDepthUpperThreshold(3000)

#     cam.preview.link(face_det_nn.input)
#     stereo.depth.link(face_det_nn.inputDepth)

#     nn_xout = pipeline.create(dai.node.XLinkOut)
#     nn_xout.setStreamName("nn_out")
#     face_det_nn.out.link(nn_xout.input)

#     if DEBUG:
#         pass_xout = pipeline.create(dai.node.XLinkOut)
#         pass_xout.setStreamName('depth')
#         face_det_nn.passthroughDepth.link(pass_xout.input)

#     print("Pipeline created.")
#     return pipeline

# def calculate_distance(coords):
#     return math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)
# def clamp(num, v0, v1):
#     return max(v0, min(num, v1))
# def get_lens_position(dist):
#     # =150-A10*0.0242+0.00000412*A10^2
#     return int(150 - dist * 0.0242 + 0.00000412 * dist**2)
# def get_lens_position_lite(dist):
#     # 141 + 0,0209x + −2E−05x^2
#     return int(141 + dist * 0.0209 - 0.00002 * dist**2)

# with dai.Device(create_pipeline()) as device:
#     controlQ = device.getInputQueue('control')


#     outputs = ['color', 'nn_out'] + (['depth'] if DEBUG else [])
#     queues: List[dai.DataOutputQueue] = [device.getOutputQueue(name, 4, False) for name in outputs]

#     sync = HostSync()
#     text = TextHelper()
#     color = (220, 220, 220)

#     lensPos = 150
#     lensMin = 0
#     lensMax = 255

#     while True:
#         for q in queues:
#             if q.has():
#                 synced_msgs = sync.add_msg(q.getName(), q.get())
#                 if synced_msgs:
#                     frame = synced_msgs["color"].getCvFrame()
#                     nn_in = synced_msgs["nn_out"]

#                     depthFrame = None # If debug
#                     if 'depth' in synced_msgs:
#                         depthFrame: dai.ImgFrame = synced_msgs["depth"].getFrame()
#                         depthFrame = cv2.pyrDown(depthFrame)
#                         depthFrame = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
#                         depthFrame = cv2.equalizeHist(depthFrame)
#                         depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_HOT)

#                     height = frame.shape[0]
#                     width  = frame.shape[1]

#                     closest_dist = 99999999
#                     for detection in nn_in.detections:
#                         # Denormalize bounding box
#                         x1 = int(detection.xmin * width)
#                         x2 = int(detection.xmax * width)
#                         y1 = int(detection.ymin * height)
#                         y2 = int(detection.ymax * height)

#                         dist = int(calculate_distance(detection.spatialCoordinates))
#                         if dist < closest_dist: closest_dist = dist
#                         text.rectangle(frame, x1,y1,x2,y2)

#                         if depthFrame is not None:
#                             roi = detection.boundingBoxMapping.roi
#                             roi = roi.denormalize(depthFrame.shape[1], depthFrame.shape[0])
#                             topLeft = roi.topLeft()
#                             bottomRight = roi.bottomRight()
#                             xmin = int(topLeft.x)
#                             ymin = int(topLeft.y)
#                             xmax = int(bottomRight.x)
#                             ymax = int(bottomRight.y)
#                             text.rectangle(depthFrame, xmin, ymin, xmax, ymax)

#                     if closest_dist != 99999999:
#                         text.putText(frame,  "Face distance: {:.2f} m".format(closest_dist/1000), (330, 1045))
#                         new_lens_pos = clamp(get_lens_position(closest_dist), lensMin, lensMax)
#                         if new_lens_pos != lensPos and new_lens_pos != 255:
#                             lensPos = new_lens_pos
#                             print("Setting manual focus, lens position: ", lensPos)
#                             ctrl = dai.CameraControl()
#                             ctrl.setManualFocus(lensPos)
#                             controlQ.send(ctrl)
#                     else:
#                         text.putText(frame,  "Face distance: /", (330, 1045))
#                     text.putText(frame, f"Lens position: {lensPos}", (330, 1000))
#                     cv2.imshow("preview", cv2.resize(frame, (750,750)))

#                     if depthFrame is not None:
#                         cv2.imshow('depth', depthFrame)

#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#         elif key in [ord(','), ord('.')]:
#             if key == ord(','): lensPos -= LENS_STEP
#             if key == ord('.'): lensPos += LENS_STEP
#             lensPos = clamp(lensPos, lensMin, lensMax)
#             print("Setting manual focus, lens position: ", lensPos)
#             ctrl = dai.CameraControl()
#             ctrl.setManualFocus(lensPos)
#             controlQ.send(ctrl)

# import cv2
# import depthai as dai
# import math
# from typing import List

# LENS_STEP = 3
# DEBUG = True

# # ----- Utility classes -----

# class TextHelper:
#     def __init__(self):
#         self.bg_color = (0, 0, 0)
#         self.color = (255, 255, 255)
#         self.text_type = cv2.FONT_HERSHEY_SIMPLEX
#         self.line_type = cv2.LINE_AA

#     def putText(self, frame, text, coords):
#         cv2.putText(frame, text, coords, self.text_type, 1.5, self.bg_color, 6, self.line_type)
#         cv2.putText(frame, text, coords, self.text_type, 1.5, self.color, 2, self.line_type)

#     def rectangle(self, frame, x1, y1, x2, y2):
#         cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 6)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)

# class HostSync:
#     def __init__(self):
#         self.arrays = {}

#     def add_msg(self, name, msg):
#         if name not in self.arrays:
#             self.arrays[name] = []
#         self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

#         synced = {}
#         for name, arr in self.arrays.items():
#             for obj in arr:
#                 if msg.getSequenceNum() == obj["seq"]:
#                     synced[name] = obj["msg"]
#                     break

#         if len(synced) == (2 if DEBUG else 1):  # Color, and Depth
#             for name in self.arrays:
#                 self.arrays[name] = [obj for obj in self.arrays[name] if obj["seq"] >= msg.getSequenceNum()]
#             return synced
#         return False

# # ----- Core functions -----

# def calculate_distance(coords):
#     return math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)

# def clamp(num, v0, v1):
#     return max(v0, min(num, v1))

# def get_lens_position(dist):
#     return int(150 - dist * 0.0242 + 0.00000412 * dist**2)

# # ----- Pipeline -----

# def create_pipeline():
#     pipeline = dai.Pipeline()

#     cam_rgb = pipeline.create(dai.node.ColorCamera)
#     cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#     cam_rgb.setPreviewSize(640, 640)
#     cam_rgb.setInterleaved(False)
#     cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#     cam_rgb.setFps(30)

#     controlIn = pipeline.create(dai.node.XLinkIn)
#     controlIn.setStreamName('control')
#     controlIn.out.link(cam_rgb.inputControl)

#     xout_video = pipeline.create(dai.node.XLinkOut)
#     xout_video.setStreamName("color")
#     cam_rgb.video.link(xout_video.input)

#     # Stereo depth
#     mono_left = pipeline.create(dai.node.MonoCamera)
#     mono_right = pipeline.create(dai.node.MonoCamera)
#     mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
#     mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
#     mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
#     mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

#     stereo = pipeline.create(dai.node.StereoDepth)
#     stereo.setConfidenceThreshold(240)
#     stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
#     stereo.setExtendedDisparity(True)

#     mono_left.out.link(stereo.left)
#     mono_right.out.link(stereo.right)

#     xout_depth = pipeline.create(dai.node.XLinkOut)
#     xout_depth.setStreamName("depth")
#     stereo.depth.link(xout_depth.input)

#     return pipeline

# # ----- Main execution -----

# with dai.Device(create_pipeline()) as device:
#     controlQ = device.getInputQueue('control')
#     queues = {name: device.getOutputQueue(name, 4, False) for name in ['color', 'depth']}
#     sync = HostSync()
#     text = TextHelper()
#     lensPos = 150
#     lensMin, lensMax = 0, 255

#     while True:
#         for name, q in queues.items():
#             if q.has():
#                 synced_msgs = sync.add_msg(name, q.get())
#                 if synced_msgs:
#                     frame = synced_msgs['color'].getCvFrame()
#                     depth_frame = synced_msgs['depth'].getFrame()

#                     # TODO: Insert your YOLO detection logic here
#                     # detections = model(frame) ...
#                     # For demo, use mock box
#                     detections = [{'bbox': (100, 100, 300, 300)}]

#                     depth_vis = cv2.pyrDown(depth_frame)
#                     depth_vis = cv2.normalize(depth_vis, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
#                     depth_vis = cv2.equalizeHist(depth_vis)
#                     depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)

#                     closest_dist = float('inf')

#                     for det in detections:
#                         x1, y1, x2, y2 = det['bbox']
#                         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#                         # Use depth value at center point
#                         depth_pixel = depth_frame[cy][cx]

#                         coords = dai.Point3f()
#                         coords.x = 0  # Estimating 2D center only
#                         coords.y = 0
#                         coords.z = depth_pixel

#                         dist = calculate_distance(coords)
#                         if dist < closest_dist:
#                             closest_dist = dist

#                         text.rectangle(frame, x1, y1, x2, y2)
#                         text.rectangle(depth_vis, x1, y1, x2, y2)

#                     if closest_dist != float('inf'):
#                         text.putText(frame, f"Distance: {closest_dist/1000:.2f} m", (30, 1045))
#                         new_lens_pos = clamp(get_lens_position(closest_dist), lensMin, lensMax)
#                         if new_lens_pos != lensPos and new_lens_pos != 255:
#                             lensPos = new_lens_pos
#                             ctrl = dai.CameraControl()
#                             ctrl.setManualFocus(lensPos)
#                             controlQ.send(ctrl)
#                     else:
#                         text.putText(frame, "Distance: /", (30, 1045))

#                     text.putText(frame, f"Lens position: {lensPos}", (30, 1000))

#                     cv2.imshow("RGB", cv2.resize(frame, (750, 750)))
#                     cv2.imshow("Depth", depth_vis)

#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#         elif key in [ord(','), ord('.')]:
#             lensPos += LENS_STEP if key == ord('.') else -LENS_STEP
#             lensPos = clamp(lensPos, lensMin, lensMax)
#             ctrl = dai.CameraControl()
#             ctrl.setManualFocus(lensPos)
#             controlQ.send(ctrl)




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
