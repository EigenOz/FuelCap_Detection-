import torch
import cv2
import depthai as dai
from transformers import AutoModelForObjectDetection, AutoImageProcessor
import numpy as np


processor = AutoImageProcessor.from_pretrained("/home/abhinav/fuel/detr-trained-20250712T131948Z-1-001/detr-trained")
model = AutoModelForObjectDetection.from_pretrained(
    "/home/abhinav/fuel/detr-trained-20250712T131948Z-1-001/detr-trained", 
    use_safetensors=True
)
model.eval()


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

     
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = processor(images=image_rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

       
        target_sizes = torch.tensor([image_rgb.shape[:2]])
        results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            box = box.detach().cpu().numpy().astype(int)
            label = label.item()
            score = score.item()
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
