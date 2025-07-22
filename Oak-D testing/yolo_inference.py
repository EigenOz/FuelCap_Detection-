import cv2
import depthai as dai
from ultralytics import YOLO

model = YOLO("/home/abhinav/fuel/fuel_yolo_training_v11_augmented-20250712T131857Z-1-001 (1)/fuel_yolo_training_v11_augmented/yolov11m_equal_split_aug/weights/best.pt")
# Add the .pt file as per your path

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam_rgb.preview.link(xout.input)


with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

    while True:
        frame = q_rgb.get().getCvFrame()  

        results = model(frame)[0] 
        annotated = results.plot()

        cv2.imshow("YOLOv11 on OAK-D Pro", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
