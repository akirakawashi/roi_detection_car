# CV_detection_car_test

CV_detection_car_test — lightweight demo for detecting vehicles inside a fixed Region Of Interest (ROI).

**Goal**

Detect vehicles only inside the area in front of a barrier using a fixed camera. Detections outside the ROI are ignored to reduce false positives and focus processing.

![ROI example](image-2.png)

## Quick Summary

- Model: pre-trained `YOLOv8X` (high-accuracy detection)
- Classes used: passenger cars and trucks
- ROI: polygon defined in the project to limit detections to the relevant area

## Requirements

- Python 3.8+
- `ultralytics`, `opencv-python`, `shapely`, `numpy`, `matplotlib`, `Pillow`

You can install typical dependencies with:

```bash
pip install ultralytics opencv-python shapely numpy matplotlib pillow
```

## Usage

- Edit the path constants in `detection.py` to point to your model and input video:

- `MODEL_PATH`, `VIDEO_INPUT_PATH`, and `VIDEO_OUTPUT_PATH`.

- Run the detection script:

```bash
python detection.py
```

The script will write the processed video to the file set in `VIDEO_OUTPUT_PATH`.

## How it works (brief)

1. Read input video frame-by-frame.
2. Run the `YOLO` model on each frame (filtered to desired classes).
3. For each detection, check if the bounding-box center lies inside the polygonal ROI.
4. Draw boxes/labels for detections inside the ROI and save frames to the output video.

## Configuration tips

- Confidence threshold: raising it (e.g. >0.6) reduces low-confidence detections and speeds up post-processing.
- If your input video resolution differs from the one used during development, update ROI coordinates accordingly (see `detection.py`).

## Development / Notebook

- Use `prediction.ipynb` for visual experiments and to preview ROI overlays on sample frames.

## Performance

- Consider processing at a lower frame rate (skip frames) if performance is a concern.
- Run inference on a GPU-supported environment (CUDA) for real-time or near-real-time processing.

## Links

- Script: ***[detection.py](detection.py)***
- Notebook: ***[prediction.ipynb](prediction.ipynb)***
- Example output video: https://rutube.ru/video/private/96861705e49012180276081d4653b0fb/?p=B67JHRbf04dCjkrF2xXqcA

If you want, I can add a minimal `requirements.txt`, a short example command with environment detection, or include sample ROI coordinates for a common resolution.  


