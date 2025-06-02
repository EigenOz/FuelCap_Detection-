# FuelCap_Detection

## The following project aims at the creation of an efficient and robust object detection system for identifying fuel caps in various environments and conditions.

This repository contains all the necessary code, datasets, training configurations, and documentation to replicate and extend the fuel cap detection solution.

---

### Table of Contents

1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Environment Setup](#environment-setup)
    * [Data Preparation](#data-preparation)
4.  [Dataset](#dataset)
    * [Structure](#structure)
    * [Acquisition and Annotation](#acquisition-and-annotation)
5.  [Model Training](#model-training)
    * [Configuration](#configuration)
    * [Training Commands](#training-commands)
6.  [Inference and Evaluation](#inference-and-evaluation)
    * [Running Inference](#running-inference)
    * [Evaluation Metrics](#evaluation-metrics)
7.  [Results](#results)
8.  [Deployment (Optional)](#deployment-optional)
9.  [Project Structure](#project-structure)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)

---

### 1. Project Overview

This project focuses on developing a deep learning model for automated fuel cap detection. The primary goal is to accurately locate and classify fuel caps in images or video streams, which can be critical for applications such as:

* Automated vehicle inspection systems.
* Robotic refueling.
* Quality control in manufacturing.
* Safety monitoring.

We leverage state-of-the-art object detection architectures (specifically YOLOv8/YOLOv11 via the `ultralytics` framework) to achieve high accuracy and inference speed.

---

### 2. Features

* **Robust Fuel Cap Detection:** Utilizes advanced YOLO models for precise localization.
* **GPU Accelerated:** Optimized for NVIDIA GPUs using PyTorch and CUDA for fast training and inference.
* **Custom Dataset:** Trained on a curated dataset of fuel caps in diverse scenarios.
* **Modular Codebase:** Easy to understand, modify, and extend for future enhancements.
* **VS Code Integration:** Optimized for development within Visual Studio Code with Jupyter Notebook support.

---

### 3. Getting Started

Follow these steps to set up the project locally and run the detection system.

#### Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Operating System:** Linux (e.g., Pop!\_OS, Ubuntu recommended for NVIDIA/ROS compatibility)
* **Python:** Version 3.8+ (preferably Python 3.10, matching your `venv` setup)
* **NVIDIA GPU:** With CUDA capabilities (check `nvidia-smi`)
* **NVIDIA Drivers:** Up-to-date NVIDIA GPU drivers compatible with your CUDA version.
* **`git`:** For cloning the repository.
* **VS Code:** Visual Studio Code for development environment.
    * **VS Code Extensions:** Python (Microsoft) and Jupyter (Microsoft) extensions.

#### Environment Setup

It is highly recommended to use a Python virtual environment (`venv`) to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/FuelCap_Detection.git](https://github.com/YourUsername/FuelCap_Detection.git)
    cd FuelCap_Detection
    ```
    (Replace `YourUsername` with your actual GitHub username if this is a new repo, or the correct organization/repo name).

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate.bat
        ```
    * **On Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```

4.  **Install `ipykernel` and register the `venv` as a Jupyter kernel:**
    *(Ensure your `venv` is activated)*
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name=fuel_cap_venv --display-name="Fuel Cap Detection VEnv"
    ```
    (You can choose a different `--name` and `--display-name` if you prefer).

5.  **Install PyTorch:**
    **Crucial:** Install PyTorch with CUDA support. Go to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and copy the `pip install` command specific to your CUDA version (e.g., CUDA 12.1).
    *(Ensure your `venv` is activated)*
    ```bash
    # Example for CUDA 12.1:
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    # If no CUDA GPU, or for CPU-only:
    # pip install torch torchvision torchaudio
    ```

6.  **Install project dependencies:**
    *(Ensure your `venv` is activated)*
    ```bash
    pip install -r requirements.txt
    ```
    *(**Note:** You'll need to create a `requirements.txt` file if you don't have one. You can generate one after installing `ultralytics` and other packages with `pip freeze > requirements.txt`)*

7.  **Configure VS Code for your `venv` (Important for ROS conflict resolution):**
    * Open the `FuelCap_Detection` folder in VS Code.
    * Create a new folder named `.vscode` at the root of the repository if it doesn't exist.
    * Inside `.vscode`, create a file named `settings.json`.
    * Add the following content to `settings.json`. **Make sure the `python3.10` path matches your system's Python version and `venv` structure.**

        ```json
        {
            "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
            "python.envFile": "${workspaceFolder}/.env",
            "terminal.integrated.env.linux": {
                "PYTHONPATH": "${workspaceFolder}/venv/lib/python3.10/site-packages:${env:PYTHONPATH}"
            },
            "jupyter.kernels.args": [
                "--JupyterLabApp.tornado_settings={'headers':{'X-Content-Type-Options': 'nosniff'}}"
            ],
            "jupyter.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python"
        }
        ```
    * **Reload VS Code (`Ctrl+Shift+P` -> `Developer: Reload Window`).**
    * In your Jupyter Notebook, **select "Fuel Cap Detection VEnv"** from the kernel dropdown in the top-right.

#### Data Preparation

*(This section will depend on how you store/access your dataset. Assuming you have a `data` directory.)*

1.  **Download / Place Dataset:**
    * Ensure your annotated dataset is placed in the `data/` directory. The expected structure is typically:
        ```
        data/
        ├── images/
        │   ├── train/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   ├── val/
        │   │   └── ...
        │   └── test/
        │       └── ...
        └── labels/
            ├── train/
            │   ├── img1.txt
            │   └── ...
            ├── val/
            │   └── ...
            └── test/
                └── ...
        ```
    * (If using a public dataset, provide download instructions/links here.)
    * (If your dataset is private, mention how others can request access or that it's not publicly available).

2.  **YAML Configuration:**
    * A `data.yaml` file (e.g., `data/fuel_cap_data.yaml`) is required for `ultralytics` to specify dataset paths, number of classes, and class names. An example structure:
        ```yaml
        # fuel_cap_data.yaml
        train: ../data/images/train
        val: ../data/images/val
        test: ../data/images/test # Optional

        # number of classes
        nc: 1

        # class names
        names: ['fuel_cap']
        ```

---

### 4. Dataset

#### Structure

The dataset follows the YOLO format for object detection, with image files and corresponding text files containing bounding box annotations.

* `images/train`, `images/val`, `images/test`: Contain the raw image files (e.g., `.jpg`, `.png`).
* `labels/train`, `labels/val`, `labels/test`: Contain the YOLO-format annotation files (`.txt`), where each line represents an object: `class_id center_x center_y width height` (normalized to image dimensions).

#### Acquisition and Annotation

*(Briefly describe your dataset's origin and how it was created/annotated.)*

The dataset was curated from various sources including [mention sources, e.g., real-world captures, online repositories, synthetic data]. Annotation was performed using [mention tool, e.g., LabelImg, Roboflow, CVAT] to mark fuel cap locations. The dataset consists of approximately [X] training images, [Y] validation images, and [Z] test images.

---

### 5. Model Training

This project utilizes the `ultralytics` library for training YOLO models.

#### Configuration

Training parameters are defined either directly in the training script or via a YAML configuration file. An example `model_config.yaml` might look like:

```yaml
# model_config.yaml
model: yolov8n.pt # or yolov8m.pt, yolov8l.pt, yolov11m.pt etc.
data: data/fuel_cap_data.yaml
epochs: 100
imgsz: 640
batch: 16
name: fuel_cap_run_v1
device: 0 # Use GPU device 0
# ... other training parameters
