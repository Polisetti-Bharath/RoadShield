# RoadShield: AI-Powered Road Damage Detection & Civic Grievance System
## Project-Based Learning (PBL) Comprehensive Documentation & Team Guide

---

## 1. Project Overview & Motivation

### 1.1 The Real-World Problem
Road infrastructure maintenance is a critical challenge across the world, especially in rapidly growing road networks like India's. Pavements deteriorate due to heavy vehicle axle loads, seasonal monsoons, waterlogging, thermal expansion, and poor drainage. Unaddressed defects such as **potholes** and **cracks** cause:
- **Severe Road Accidents & Fatalities**: Tens of thousands of road accidents each year are directly linked to unexpected potholes and broken pavement.
- **Economic Loss**: Vehicle wear-and-tear (suspension failure, rim fractures, tire punctures) and logistics delays cost billions of rupees annually.
- **Exponential Repair Costs**: Repairing a neglected road that has structurally failed costs **4 to 5 times more** than early preventive maintenance (filling cracks before they expand into deep craters).

### 1.2 Traditional vs. RoadShield Approach
* **Traditional Method**: Municipal inspectors physically travel roads, take manual notes, and fill paper forms. This is slow, subjective, unsafe for personnel, covers only a tiny percentage of roads, and rarely translates into immediate action.
* **RoadShield (Our Solution)**: An end-to-end AI-powered vision system that can be mounted on vehicles (dashcams, municipal garbage trucks, police cruisers) or used on mobile phones. RoadShield automatically **detects, categorizes, and measures road damage**, extracts **GPS coordinates**, converts them to a **human-readable address**, estimates **hazard severity**, and **generates official civic complaints** via WhatsApp, Email, or downloadable PDF for portals like **CPGRAMS** (Centralized Public Grievance Redress and Monitoring System) or municipal corporation apps (e.g., BBMP Sahaaya).

---

## 2. Target Road Damage Classes

RoadShield uses the international pavement classification defined in the **Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)**:

| Class Code | Damage Name | Visual Pattern | Cause & Structural Impact | UI Badge Color |
| :---: | :--- | :--- | :--- | :---: |
| **D00** | **Longitudinal Crack** | Single line running parallel to traffic direction | Joint fatigue, paving seam weakness, or subgrade settlement. | Amber / Orange |
| **D10** | **Transverse Crack** | Single line running across the road (perpendicular) | Temperature cycles, asphalt shrinkage, or base layer cracking. | Cyan / Blue |
| **D20** | **Alligator Crack** | Interconnected web of cracks resembling reptile skin | Severe structural base failure from repetitive heavy wheel loads. | Red / Crimson |
| **D40** | **Potholes** | Bowl-shaped cavity/depression in road surface | Water seeped under asphalt freezes/erodes base, dislodging asphalt pieces. | Purple / Indigo |

---

## 3. System Architecture & How It Works

RoadShield is designed around a 4-stage pipeline: **Ingest $\rightarrow$ Infer $\rightarrow$ Assess $\rightarrow$ Act**.

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion Layer"]
        A1["📷 Live Camera Feed<br>(Webcam / Dashcam via WebRTC)"]
        A2["🖼️ Static Image Upload<br>(JPG / PNG + EXIF Metadata)"]
        A3["🎬 Video Footage<br>(MP4 Dashcam Recording)"]
    end

    subgraph CoreVision["2. Deep Learning Inference Layer"]
        B["YOLOv8-Small Model<br>(Trained on CRDDC2022 Japan + India)"]
        B -->|Bounding Box Coordinates [x1, y1, x2, y2]| C["Class Prediction & Confidence Score"]
    end

    subgraph Assessment["3. Assessment & Geolocation Layer"]
        D1["Severity Estimator<br>(Damage Area / Frame Area %)"]
        D2["GPS Extraction<br>(piexif from Image / Browser Geolocation API)"]
        D3["Reverse Geocoding<br>(OpenStreetMap Nominatim API)"]
    end

    subgraph CivicReporting["4. Civic Action & Reporting Layer"]
        E1["💬 Instant WhatsApp Complaint<br>(Pre-filled text & Google Maps pin)"]
        E2["✉️ Municipal Email Dispatch<br>(Detailed subject, coordinates, & description)"]
        E3["📄 Official PDF Complaint Card<br>(FPDF2 formatted with annotated image for CPGRAMS)"]
    end

    A1 --> B
    A2 --> B
    A3 --> B
    C --> D1
    A2 --> D2
    A1 --> D2
    D2 --> D3
    D1 --> CivicReporting
    D3 --> CivicReporting
```

### 3.1 The Three Inspection Modes
1. **Live Realtime Detection (`pages/1_Realtime_Detection.py`)**:
   - Uses `streamlit-webrtc` to stream browser video over WebRTC with ultra-low latency.
   - An `av.VideoFrame` callback transforms each frame: converts the raw video packet to a BGR NumPy array, passes it to the cached `YOLO` instance, draws color-coded bounding boxes and labels using OpenCV, and emits the annotated frame back to the browser.
   - Leverages STUN (Session Traversal Utilities for NAT) servers (`stun.l.google.com:19302`) with fallback logic to ensure connectivity even across firewalls and cellular networks.
   - Provides live detection counters in a table beneath the video player.

2. **Static Image Detection (`pages/2_Image_Detection.py`)**:
   - Users upload standard road photos (e.g., captured by pedestrians or municipal field workers) or test with built-in sample images.
   - Automatically parses **EXIF metadata** using `piexif` to extract GPS coordinates embedded by smartphones.
   - Lets users adjust a real-time **Confidence Threshold Slider** (0.05 to 1.00) to filter detections.
   - Runs inference, overlays bounding boxes with custom class colors, and outputs side-by-side or overlaid visuals with class frequency counters.

3. **Offline Video Detection (`pages/3_Video_Detection.py`)**:
   - Designed for municipal vehicles (garbage trucks, buses, patrol vehicles) carrying dashcams.
   - Accepts uploaded `.mp4` drive-through recordings.
   - Decodes video frame-by-frame using OpenCV `cv2.VideoCapture`.
   - Runs YOLOv8 inference across frames with an interactive progress bar showing elapsed percentage and estimated time remaining.
   - Re-encodes the annotated frames into an MP4 video stream using OpenCV's `VideoWriter` and presents the final processed video for immediate playback and download.

### 3.2 Automated Severity Estimation
RoadShield computes defect severity dynamically using the **bounding box coverage ratio**:
$$\text{Damage Ratio} (\%) = \frac{\sum (x_2 - x_1) \times (y_2 - y_1)}{W_{\text{frame}} \times H_{\text{frame}}} \times 100$$
- **Low Severity**: $< 2\%$ frame coverage (incipient cracks).
- **Medium Severity**: $2\% - 6\%$ frame coverage (developing potholes or prominent cracks).
- **High Severity**: $\ge 6\%$ frame coverage (massive crater potholes or heavy alligator cracking demanding urgent intervention).

### 3.3 Geolocation & Civic Action Pipeline
1. **Coordinate Acquisition**:
   - If an uploaded photo contains EXIF GPS tags, coordinates are extracted automatically.
   - If not (or in live/video modes), the user clicks "Share Location" powered by HTML5 browser geolocation (`streamlit-geolocation`), or inputs coordinates manually.
2. **Reverse Geocoding**:
   - Calls the OpenStreetMap **Nominatim API** (`https://nominatim.openstreetmap.org/reverse`) to convert latitude/longitude into human-readable street, district, and state addresses (cached for 1 hour to prevent redundant API queries).
3. **Dispatch Channels**:
   - **WhatsApp**: Generates a pre-formatted `https://wa.me/<number>?text=...` deep link containing damage type, severity, address, and Google Maps pin.
   - **Email**: Generates a `mailto:` link with standardized grievance text ready for civic bodies (e.g., PWD, Municipal Commissioner).
   - **Formal PDF Generation (`fpdf2`)**: Generates an official single-page complaint PDF featuring the annotated road evidence image, exact timestamp, coordinates, severity classification, and clickable Google Maps link, tailored for attachment to portals like **CPGRAMS** (`pgportal.gov.in`).

---

## 4. Complete Technology Stack

| Layer / Domain | Technology / Library | Version / Tool | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **Programming Language** | Python | `3.10` / `3.11` | Core development language for scripts, pipeline, and web app. |
| **Deep Learning Framework** | Ultralytics YOLOv8 | `ultralytics 8.4.157` | State-of-the-art anchor-free object detection model (`YOLOv8-Small`). |
| **Tensor Computation** | PyTorch & Torchvision | `torch`, `torchvision` (CUDA/CPU) | Deep learning runtime, tensor calculations, and GPU acceleration. |
| **Computer Vision** | OpenCV | `opencv-python-headless 5.0.0.93` | Video decoding, frame extraction, bounding box rendering, image transformations. |
| **Image Processing** | Pillow | `Pillow 12.3.0` | High-fidelity image manipulation, format conversions, and canvas rendering. |
| **Frontend Framework** | Streamlit | `streamlit 1.64.0` | Multi-page reactive web application framework. |
| **Real-time Video** | streamlit-webrtc & PyAV | `streamlit-webrtc 0.78.1`, `av 17.1.0` | Low-latency WebRTC browser camera streaming and frame transformation callback. |
| **Network Traversal** | STUN Protocol | `stun.l.google.com:19302` | NAT traversal protocol allowing WebRTC video across diverse router/firewall networks. |
| **EXIF Parsing** | piexif | `piexif 1.1.3` | Parsing embedded smartphone GPS metadata directly from JPEG image headers. |
| **Browser Geolocation** | streamlit-geolocation | `streamlit-geolocation 0.0.10` | HTML5 client-side GPS location capture directly from user's device. |
| **Reverse Geocoding** | Requests & OSM Nominatim | `requests 2.34.2` | Translates raw GPS lat/lon into human-readable road and neighborhood addresses. |
| **Document Generation** | FPDF2 | `fpdf2 2.8.8` | Generates official, tamper-evident PDF complaint reports for municipal submissions. |
| **UI Design System** | Custom Vanilla CSS | `sample_utils/ui.py` | Glassmorphic cards, glowing badges, dark mode aesthetic, responsive metric grids. |
| **Environment & Package Mgmt** | uv / pip | `uv` virtual environment | High-speed dependency isolation and deterministic package management. |

---

## 5. Three-Person Contribution Division

To ensure an equal, clearly delineated, and professional distribution of work for Project-Based Learning (PBL) submission and evaluation, the project is divided into three distinct roles:

```
+---------------------------------------------------------------------------------------------------+
|                                  ROADSHIELD TEAM DIVISION                                         |
+------------------------------------+----------------------------------+---------------------------+
| Member 1: AI/ML & Dataset Lead     | Member 2: App & WebRTC Engineer  | Member 3: Geospatial &    |
|                                    |                                  | Civic Integration Lead    |
+------------------------------------+----------------------------------+---------------------------+
| • CRDDC2022 Dataset Preparation    | • Streamlit Multi-Page App       | • EXIF GPS & HTML5 Geo    |
| • PascalVOC -> YOLO TXT Conversion | • Modern CSS Glassmorphic UI     | • OSM Nominatim Geocoding |
| • Background Image Filtering       | • Real-Time WebRTC Video Pipeline| • Severity Area Algorithm |
| • YOLOv8s Model Training & Tuning  | • Static Image Inference Flow    | • WhatsApp / Email Links  |
| • Metrics, PR-Curves & Confusion M.| • Batch Video Processing Engine  | • FPDF2 Grievance PDFs    |
+------------------------------------+----------------------------------+---------------------------+
```

---

### Member 1: Machine Learning & Computer Vision Lead
* **Primary Role**: Data engineering, deep learning model architecture, training, hyperparameter optimization, and statistical validation.
* **Key Responsibilities**:
  1. **Dataset Acquisition & Preprocessing**:
     - Curated the CRDDC2022 multi-country benchmark, selecting road condition data from **India** and **Japan** to simulate real Asian driving conditions.
     - Authored `training/0_PrepareDatasetYOLOv8.ipynb` to parse Pascal VOC XML annotation files (`<bndbox>`, `<xmin>`, `<ymin>`, `<xmax>`, `<ymax>`).
     - Converted bounding box pixel coordinates into YOLO-normalized format:
       $$x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2 \cdot H}, \quad w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$
     - Implemented **background filtering** to clean out excess non-annotated background images, avoiding dataset dilution.
     - Engineered a clean 80/20 train/validation split and created the Ultralytics dataset configuration file `rdd_JapanIndia.yaml`.
  2. **Model Training & Hyperparameter Tuning**:
     - Handled `training/1_TrainingYOLOv8.ipynb`. Selected `YOLOv8-Small` (`yolov8s.pt`) as the optimal balance between high frame-rate edge inference and feature extraction capacity.
     - Configured hyperparameter training runs (batch size, input resolution $640 \times 640$, learning rate scheduling, Mosaic augmentation).
     - Generated and saved the production checkpoint `models/YOLOv8_Small_RDD.pt`.
  3. **Model Evaluation & Performance Benchmarking**:
     - Authored `training/2_EvaluationTesting.ipynb` to run validation on held-out test splits.
     - Computed evaluation metrics: Precision, Recall, $\text{mAP}@0.5$, and $\text{mAP}@0.5:0.95$.
     - Generated and analyzed validation artifacts: Precision-Recall curves (`resource/PR_curve.png`), Confusion Matrix (`resource/confusion_matrix.png`), and prediction batch visualizations (`resource/val_batch2_pred.jpg`).
* **Specific Files Owned**:
  - `training/0_PrepareDatasetYOLOv8.ipynb`
  - `training/1_TrainingYOLOv8.ipynb`
  - `training/2_EvaluationTesting.ipynb`
  - `models/YOLOv8_Small_RDD.pt`
  - `training/yolov8s.pt`, `training/yolov8n.pt`
  - Performance curves in `resource/`
* **Viva / Defense Talking Points**:
  - *"Why YOLOv8 over YOLOv5 or Faster R-CNN?"* $\rightarrow$ YOLOv8 uses an **anchor-free split head**, eliminating predefined anchor box tuning and reducing false positives on irregular, thin objects like cracks. It also replaces C3 modules with **C2f** (cross-stage partial with multiple gradient flows), leading to faster convergence and better gradient backpropagation.
  - *"How did you tackle class imbalance?"* $\rightarrow$ Filtered zero-label images to increase positive sample density and leveraged YOLOv8's Task-Aligned Assigner and Distribution Focal Loss (DFL).

---

### Member 2: Frontend, Web Architecture & Real-Time Media Engineer
* **Primary Role**: Full-stack application architecture, reactive UI/UX design, real-time video streaming, and media inference orchestration.
* **Key Responsibilities**:
  1. **Application Architecture & Navigation**:
     - Built the multi-page Streamlit application structure: `Home.py` as landing portal and dedicated subpages in `pages/` for isolated functional flows.
     - Configured system settings in `.streamlit/config.toml` for optimized server execution and clean layout.
  2. **Design System & Aesthetics (`sample_utils/ui.py`)**:
     - Designed a custom CSS styling layer injected dynamically into Streamlit (`inject_base_css()`).
     - Built modern glassmorphic cards (`.rs-feature-card`), interactive glowing borders (`glow-indigo`, `glow-violet`, `glow-pink`), responsive metric grids, and custom typography using the Inter font family.
     - Created color-coded UI badges (`CLASS_COLORS`) matching the 4 road distress categories for consistent visual hierarchy across pages.
  3. **Real-Time WebRTC Streaming (`pages/1_Realtime_Detection.py`)**:
     - Integrated `streamlit-webrtc` and `PyAV` to stream client webcam frames to the Python backend without server browser overhead.
     - Implemented the custom `VideoTransformer` / callback function that performs frame unpacking, runs YOLO inference per frame, overlays OpenCV visual annotations, and sends the processed frame back to the client at interactive frame rates.
  4. **Image & Video Media Pipelines (`pages/2_Image_Detection.py`, `pages/3_Video_Detection.py`)**:
     - Implemented image upload, PIL format conversion, OpenCV bounding box rendering with alpha blending, and dynamic confidence threshold filtering.
     - Built the offline batch video processing engine using OpenCV `VideoCapture` and `VideoWriter`, including progress tracking, frame counting, elapsed time calculation, and final annotated video rendering.
* **Specific Files Owned**:
  - `Home.py`
  - `pages/1_Realtime_Detection.py`
  - `pages/2_Image_Detection.py`
  - `pages/3_Video_Detection.py`
  - `sample_utils/ui.py`
  - `.streamlit/config.toml`
* **Viva / Defense Talking Points**:
  - *"Why WebRTC instead of standard `cv2.VideoCapture(0)`?"* $\rightarrow$ Standard `cv2.VideoCapture(0)` accesses the server's local camera hardware, which fails entirely when deployed to a remote server or accessed over a network. WebRTC streams the client's browser camera over peer-to-peer RTP/SRTP channels, making the app genuinely cross-platform and cloud-ready.
  - *"How did you maintain performance in video processing?"* $\rightarrow$ Leveraged PyTorch model caching via `@st.cache_resource` so the model weights load once in memory, batching OpenCV frame operations and using headless OpenCV to eliminate display buffer bottlenecks.

---

### Member 3: Geospatial, Civic Action & Reporting Integration Lead
* **Primary Role**: Metadata extraction, geospatial processing, automated municipal reporting, and civic grievance integration.
* **Key Responsibilities**:
  1. **Geospatial & Metadata Extraction**:
     - Implemented `get_gps_from_exif()` in `sample_utils/report.py` using `piexif` to convert Degrees-Minutes-Seconds (DMS) GPS tags from smartphone photos into decimal coordinates:
       $$\text{Decimal Degrees} = \text{Degrees} + \frac{\text{Minutes}}{60} + \frac{\text{Seconds}}{3600}$$
     - Integrated client-side HTML5 browser geolocation via `streamlit-geolocation` with a manual coordinate fallback popover for devices without GPS.
  2. **Reverse Geocoding Engine**:
     - Implemented `reverse_geocode()` connecting to the OpenStreetMap **Nominatim API** with custom User-Agent headers, converting latitude/longitude into formatted residential addresses, roads, and cities.
     - Added `@st.cache_data(ttl=3600)` to cache coordinate queries, preventing rate limits and ensuring sub-second UI response times.
  3. **Damage Severity Estimation Algorithm**:
     - Authored `estimate_severity()` calculating the percentage of frame area occupied by the bounding box, classifying defects into **Low**, **Medium**, and **High** priority levels to assist municipal road engineers in prioritizing emergency patching.
  4. **Civic Action Dispatch & PDF Generator**:
     - Built the automated one-click **WhatsApp grievance generator** with URL-encoded messages including damage type, severity, location, and Google Maps pin.
     - Built the **Email complaint dispatch generator** with pre-filled subject line and body text.
     - Developed `build_complaint_pdf()` using `fpdf2`, creating professional single-page damage reports containing the photographic evidence, date/time, coordinates, severity grade, and clickable map links formatted for submission on portals like **CPGRAMS** (`pgportal.gov.in`) or city apps.
  5. **Network Utilities (`sample_utils/get_STUNServer.py`, `sample_utils/download.py`)**:
     - Handled STUN server resolution for WebRTC NAT traversal and utilities for model/sample downloads.
* **Specific Files Owned**:
  - `sample_utils/report.py`
  - `sample_utils/get_STUNServer.py`
  - `sample_utils/download.py`
  - `.rs_authority_contact.json` persistence handling
* **Viva / Defense Talking Points**:
  - *"How do you turn computer vision into real-world civic impact?"* $\rightarrow$ Detecting damage is only half the battle; municipal bodies require structured evidence with verifiable location data. RoadShield bridges this gap by automatically attaching verified GPS coordinates, reverse-geocoded addresses, and generating official complaint PDFs and instant WhatsApp messages ready for municipal portals like CPGRAMS or municipal lines.
  - *"How does EXIF extraction work?"* $\rightarrow$ Smartphones encode EXIF IFD GPS tags in the image header. `piexif` parses binary tags `GPSLatitude`, `GPSLongitude`, and their respective reference directions (`N/S`, `E/W`), converting sexagesimal DMS tuples into standard WGS84 decimal coordinates.

---

## 6. Project Directory Structure

```
RoadShield/
├── Home.py                             # Main Streamlit landing page & navigation
├── pages/
│   ├── 1_Realtime_Detection.py        # WebRTC live webcam streaming & inference
│   ├── 2_Image_Detection.py           # Photo upload, EXIF GPS reading & detection
│   └── 3_Video_Detection.py           # Batch offline MP4 video processing
├── sample_utils/
│   ├── ui.py                          # Modern CSS design system, cards, colors & badges
│   ├── report.py                      # EXIF GPS, reverse geocoding, severity & PDF engine
│   ├── model.py                       # Deduplicated YOLO model loader with caching
│   ├── logging_config.py              # Structured application logging
│   └── get_STUNServer.py              # STUN server configuration for WebRTC NAT traversal
├── models/
│   └── YOLOv8_Small_RDD.pt            # Production trained weights (CRDDC2022 Japan + India)
├── tests/                             # Automated test suite (Pytest)
│   ├── test_get_STUNServer.py
│   ├── test_logging_config.py
│   ├── test_model.py
│   └── test_report.py
├── .github/workflows/                  # Enterprise CI/CD pipelines
│   ├── ci.yml                         # Lint (ruff), tests (pytest), image build
│   ├── cd.yml                         # Gated release pipeline & GHCR publishing
│   ├── security.yml                   # Scheduled pip-audit & Trivy vulnerability scans
│   └── dependabot.yml                 # Automated weekly dependency updates
├── training/
│   ├── 0_PrepareDatasetYOLOv8.ipynb   # Pascal VOC XML to YOLO TXT dataset preprocessing
│   ├── 1_TrainingYOLOv8.ipynb         # Ultralytics YOLOv8s training script
│   ├── 2_EvaluationTesting.ipynb      # Metrics calculation & test inference
│   └── yolov8s.pt                     # Base Small weights
├── resource/                          # Evaluation plots, sample images, GIFs
├── Dockerfile                         # Production container definition
├── docker-compose.yml                 # Container orchestration
├── pyproject.toml                     # Ruff linter and Pytest configuration
├── requirements.txt                   # Production dependencies with security pins
├── requirements-dev.txt               # Development & test dependencies
├── packages.txt                       # Linux OS-level dependencies (libgl1)
└── pbl.md                             # Comprehensive PBL Documentation (This File)
```

---

## 7. Setup & Execution Guide

### 7.1 Prerequisites
- Python **3.10** or **3.11** installed.
- (Optional) NVIDIA GPU with CUDA 12.1 for accelerated training/inference.

### 7.2 Installation Commands

```bash
# 1. Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. If using an NVIDIA GPU, install PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# If CPU-only:
# pip install torch torchvision

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run Home.py
```

### 7.3 How to Demo for Evaluation / Presentation
1. **Landing Page (`Home.py`)**:
   - Open browser at `http://localhost:8501`.
   - Point out the 4 damage categories and modern dark-mode card interface.
2. **Image Detection Demo (`pages/2_Image_Detection.py`)**:
   - Upload any sample road image or click one of the preset sample damage buttons.
   - Adjust the **Confidence Threshold** slider to show how predictions filter out noise.
   - Demonstrate the **Severity Rating** (Low/Medium/High).
   - Show the **Civic Action Card**: show how the address is resolved from GPS, click **"Download PDF Report"** to show the generated grievance document, and show the WhatsApp / Email links.
3. **Realtime Detection Demo (`pages/1_Realtime_Detection.py`)**:
   - Click "START" on the WebRTC camera widget to show live bounding boxes and detection counts.
4. **Video Processing Demo (`pages/3_Video_Detection.py`)**:
   - Upload a road clip to show frame-by-frame progress and playback of the annotated result.

---

## 8. Faculty Viva Q&A Cheatsheet

### Q1: Why did you choose YOLOv8 instead of older architectures like YOLOv5 or Faster R-CNN?
* **Answer**: YOLOv8 is an **anchor-free** detector, meaning it predicts the bounding box center and offsets directly instead of relying on manually tuned anchor boxes. Road cracks are non-standard, elongated, and irregular, making predefined anchor boxes ineffective. In addition, YOLOv8 replaces the C3 module with **C2f** (cross-stage partial network with multiple gradient paths), improving lightweight feature extraction, and uses a decoupled head separating classification and bounding box regression tasks for faster convergence.

### Q2: What dataset was used and how did you preprocess it?
* **Answer**: We used the **Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)** dataset, focusing on the **India** and **Japan** subsets. Raw annotations were in Pascal VOC XML format with pixel coordinates. We wrote a preprocessing notebook (`0_PrepareDatasetYOLOv8.ipynb`) that converted XML coordinates into normalized YOLO format $(x_c, y_c, w, h)$, filtered out excess negative (background) images, split the data into 80% training and 20% validation subsets, and generated the dataset configuration file `rdd_JapanIndia.yaml`.

### Q3: How is the severity of the damage determined?
* **Answer**: Severity is computed mathematically by measuring the proportion of the frame occupied by the defect bounding box: $\frac{\text{Box Area}}{\text{Frame Area}} \times 100$. If the defect covers less than 2% of the frame, it is categorized as **Low** (routine crack); between 2% and 6% as **Medium**; and 6% or greater as **High** (major pothole or large structural failure requiring urgent priority).

### Q4: How does the real-time webcam feed work without crashing the browser?
* **Answer**: We use `streamlit-webrtc`, which establishes a **WebRTC peer connection** between the client browser and the server. Video frames are streamed as `av.VideoFrame` objects over UDP/SRTP. A custom Python callback intercepts each frame, converts it into a NumPy array, runs cached YOLO inference, overlays the annotations using OpenCV, and returns the modified frame back to the browser with minimal latency.

### Q5: How does the application locate the road damage?
* **Answer**: In image detection mode, RoadShield inspects the photo's EXIF metadata using the `piexif` library. If the image was captured on a smartphone with location enabled, the latitude and longitude are extracted from the GPS IFD tags. In live or video mode, the user can grant permission via the HTML5 browser geolocation API (`streamlit-geolocation`). The coordinates are then passed to the OpenStreetMap **Nominatim reverse geocoding API** to obtain the street, neighborhood, and city name.

### Q6: What is the practical civic utility of RoadShield?
* **Answer**: Existing road inspection systems either stop at detection or require proprietary hardware. RoadShield bridges inspection and citizen/municipal governance. It automates the generation of compliant complaint documentation: an instant WhatsApp notification with coordinates and Google Maps pin, an automated email, and an official PDF evidence report compatible with portals like **CPGRAMS** (`pgportal.gov.in`) or city municipal bodies.

---

## 9. Summary & Future Scope

### 9.1 Key Highlights of the Project
- **End-to-End Pipeline**: Combines deep learning object detection, real-time media streaming, geospatial analysis, and civic governance.
- **Production-Ready UI**: Custom glassmorphic design system that avoids default unstyled components.
- **Multimodal Flexibility**: Works on live webcam streams, uploaded static photos, and batch recorded dashcam videos.
- **Civic Automation**: Directly addresses public road safety by generating ready-to-file grievance reports.

### 9.2 Future Enhancements
- **Edge Deployment**: Porting the model to TensorRT or ONNX Runtime on edge hardware such as Raspberry Pi 5 or NVIDIA Jetson Orin Nano for in-vehicle black-box deployment.
- **Depth Estimation**: Integrating monocular depth estimation (e.g., MiDaS or Depth Anything) to measure the physical depth and volumetric displacement of potholes in centimeters.
- **Municipal Dashboard**: Developing a centralized GIS map interface (e.g., Leaflet or Mapbox) where city authorities can view a heatmap of all reported potholes and assign work orders to contractors.
