# Video Inferencing with OAK-D Pro

Leverage your OAK-D Pro camera for fast object detection and depth estimation using YOLO and DeTR models. Below are instructions for running each model, important performance notes, and usage tips.

## Requirements

- OAK-D Pro camera
- Python 3.8 or later
- Dependencies from `requirements.txt`
- Pretrained model weights (YOLO or DeTR)
- (For DeTR) Preprocessor matching your safetensors file

## YOLO Model Inference

**Scripts:**
- `yolo_inference.py`: Runs YOLO object detection on live video.
- `object_depth_estimation.py`: Computes and prints the distance to detected objects.

**Usage:**
- Real-time object detection (~50 FPS expected).

- Prints real-time distance estimates for each detected object.

## DeTR Model Inference

**Script:**
- `detr_inference.py`: Runs DeTR (Detection Transformer) model on OAK-D Pro video stream.

**Usage:**
- Expected performance: ~7 FPS.

> **Note:**  
> Ensure you use the preprocessor that matches the DeTR safetensors file you are loading. A mismatched preprocessor can cause invalid detections.

## Additional Notes

- Scripts can be adapted for different model versions.
- Ensure the camera is connected and all dependencies are installed.
- Actual FPS may vary based on your hardware and model.

## Troubleshooting

- **Low FPS:** Close other apps and use optimized models if available.
- **Model/preprocessor issues:** Double-check that you're using the correct preprocessor and model weights.

For more information, consult the OAK-D Pro SDK or your chosen model’s documentation.
