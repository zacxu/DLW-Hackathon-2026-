import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def preprocess_video(video_path: str, sequence_length: int, image_height: int, image_width: int) -> np.ndarray:
    """Extract evenly spaced frames and normalize to [0, 1]."""
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)

    if not video_reader.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (image_width, image_height))
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()

    if len(frames_list) != sequence_length:
        raise ValueError(
            f"Video does not have enough readable frames for preprocessing. "
            f"Expected {sequence_length}, got {len(frames_list)}."
        )

    return np.asarray(frames_list, dtype=np.float32)


def preprocess_image_as_sequence(
    image_path: str, sequence_length: int, image_height: int, image_width: int
) -> np.ndarray:
    """Preprocess a single image and repeat it across the sequence dimension."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    resized_image = cv2.resize(image, (image_width, image_height))
    normalized_image = resized_image.astype(np.float32) / 255.0
    return np.repeat(normalized_image[np.newaxis, ...], sequence_length, axis=0)


def preprocess_violence_input(
    input_path: str, sequence_length: int, image_height: int, image_width: int
) -> np.ndarray:
    source_path = Path(input_path)
    if is_video_file(source_path):
        return preprocess_video(
            video_path=input_path,
            sequence_length=sequence_length,
            image_height=image_height,
            image_width=image_width,
        )
    if is_image_file(source_path):
        return preprocess_image_as_sequence(
            image_path=input_path,
            sequence_length=sequence_length,
            image_height=image_height,
            image_width=image_width,
        )
    raise ValueError(f"Unsupported input type for violence model: {source_path}")


def run_violence_detection(input_path: str, model_path: str = "models/violence_MobileNet.keras") -> float:
    """Load violence model, preprocess video/image using model input shape, and return confidence."""
    model = tf.keras.models.load_model(model_path)

    input_shape = model.input_shape
    if len(input_shape) != 5:
        raise ValueError(f"Expected 5D model input (batch, time, h, w, c), got: {input_shape}")

    _, sequence_length, image_height, image_width, channels = input_shape
    if channels != 3:
        raise ValueError(f"Expected 3-channel RGB/BGR input, got channels={channels}")

    preprocessed_frames = preprocess_violence_input(
        input_path=input_path,
        sequence_length=int(sequence_length),
        image_height=int(image_height),
        image_width=int(image_width),
    )

    model_input = np.expand_dims(preprocessed_frames, axis=0)
    prediction = model.predict(model_input, verbose=0)
    confidence = float(np.squeeze(prediction))
    return confidence


def extract_frame_for_detection(video_path: str) -> np.ndarray:
    """Extract a representative frame from the middle of a video."""
    video_reader = cv2.VideoCapture(video_path)
    if not video_reader.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = max(total_frames // 2, 0)
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    success, frame = video_reader.read()
    if not success:
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = video_reader.read()

    video_reader.release()

    if not success:
        raise ValueError(f"Unable to read a frame from video: {video_path}")

    return frame


def is_video_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in VIDEO_SUFFIXES


def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_SUFFIXES


def parse_yolo_result(result) -> list[dict]:
    """Convert a YOLO result object into structured detections."""
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xywhn = result.boxes.xywhn.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    names = result.names

    for cls_id, box, conf in zip(class_ids, xywhn, confidences):
        class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        detections.append(
            {
                "class_name": str(class_name),
                "confidence": float(conf),
                "bbox_xywhn": box.tolist(),
            }
        )

    return detections


def extract_fire_confidence(detections: list[dict]) -> float:
    """Get fire confidence from detections. Returns 0.0 if fire is not detected."""
    fire_scores = [
        det["confidence"]
        for det in detections
        if "fire" in det["class_name"].strip().lower()
    ]
    return float(max(fire_scores)) if fire_scores else 0.0


def run_fire_detection(input_path: str, yolo_model_path: str = "models/fire.pt") -> float:
    """Run YOLO on an image or representative video frame and return fire confidence."""
    source_path = Path(input_path)
    yolo_model = YOLO(yolo_model_path)

    if is_video_file(source_path):
        frame = extract_frame_for_detection(str(source_path))
        result = yolo_model.predict(source=frame, verbose=False)[0]
    elif is_image_file(source_path):
        result = yolo_model.predict(source=str(source_path), verbose=False)[0]
    else:
        raise ValueError(f"Unsupported input type for fire model: {source_path}")

    detections = parse_yolo_result(result)
    return extract_fire_confidence(detections)


def run_all(
    input_path: str,
    violence_model_path: str = "models/violence_MobileNet.keras",
    yolo_model_path: str = "models/fire.pt",
) -> list[dict]:
    """
    Run all implemented anomaly models on an image/video and return JSON-ready results.
    """
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")
    if not is_video_file(source_path) and not is_image_file(source_path):
        raise ValueError(f"run_all requires an image or video input, got: {source_path}")

    violence_confidence = run_violence_detection(str(source_path), violence_model_path)
    fire_confidence = run_fire_detection(str(source_path), yolo_model_path)

    return [
        {
            "detected_anomaly": "violence",
            "model_used": str(Path(violence_model_path).name),
            "confidence": float(violence_confidence),
        },
        {
            "detected_anomaly": "fire",
            "model_used": str(Path(yolo_model_path).name),
            "confidence": float(fire_confidence),
        },
    ]


def run_yolo_detection(
    input_path: str,
    yolo_model_path: str = "models/fire.pt",
    output_dir: str = "outputs",
    visualization_root: str = "datasets/fire",
) -> Path:
    """
    Run YOLO on an input image or representative video frame and save annotated output image.
    """
    import visualization as visualization_module
    from visualization import Visualization

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_path = Path(input_path)
    if is_video_file(source_path):
        frame = extract_frame_for_detection(str(source_path))
        frame_path = output_path / f"{source_path.stem}_frame.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            raise ValueError(f"Unable to save extracted frame to: {frame_path}")
        yolo_source_path = frame_path
    elif is_image_file(source_path):
        yolo_source_path = source_path
    else:
        raise ValueError(f"Unsupported input type for YOLO: {source_path}")

    yolo_model = YOLO(yolo_model_path)
    results = yolo_model.predict(source=str(yolo_source_path), verbose=False)
    result = results[0]

    bboxes = []
    for det in parse_yolo_result(result):
        x_center, y_center, width, height = det["bbox_xywhn"]
        bboxes.append(
            [
                det["class_name"],
                x_center,
                y_center,
                width,
                height,
                det["confidence"],
            ]
        )

    visualization_module.root = str(visualization_root)
    vis = Visualization(root=str(visualization_root), data_types=[], n_ims=1, rows=1, cmap="rgb")

    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 8))
    vis.plot(1, 1, 1, str(yolo_source_path), bboxes)
    annotated_path = output_path / f"{source_path.stem}_yolo_bbox.jpg"
    plt.tight_layout()
    plt.savefig(str(annotated_path), bbox_inches="tight")
    plt.close()

    return annotated_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run violence/fire detection on image or video inputs."
    )
    parser.add_argument("input_path", type=str, help="Path to input video or image file.")
    parser.add_argument(
        "--violence-model-path",
        type=str,
        default="models/violence_MobileNet.keras",
        help="Path to Keras violence model file (default: models/violence_MobileNet.keras).",
    )
    parser.add_argument(
        "--run-yolo",
        action="store_true",
        help="Run YOLO detection and save an annotated image to outputs.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all implemented models on image/video input and print JSON output.",
    )
    parser.add_argument(
        "--yolo-model-path",
        type=str,
        default="models/fire.pt",
        help="Path to YOLO model file (default: models/fire.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store YOLO output images (default: outputs).",
    )
    parser.add_argument(
        "--visualization-root",
        type=str,
        default="datasets/fire",
        help="Root path containing data.yaml used by Visualization class (default: datasets/fire).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    violence_model_path = Path(args.violence_model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.run_all:
        yolo_model_path = Path(args.yolo_model_path)
        if not violence_model_path.exists():
            raise FileNotFoundError(f"Violence model file not found: {violence_model_path}")
        if not yolo_model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {yolo_model_path}")

        all_results = run_all(
            input_path=str(input_path),
            violence_model_path=str(violence_model_path),
            yolo_model_path=str(yolo_model_path),
        )
        print(json.dumps(all_results, indent=2))
        return

    if not violence_model_path.exists():
        raise FileNotFoundError(f"Violence model file not found: {violence_model_path}")
    if not is_video_file(input_path) and not is_image_file(input_path):
        raise ValueError(f"Unsupported input type: {input_path}")

    confidence = run_violence_detection(str(input_path), str(violence_model_path))
    print(f"Confidence score: {confidence:.6f}")

    if args.run_yolo:
        yolo_model_path = Path(args.yolo_model_path)
        if not yolo_model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {yolo_model_path}")

        vis_root = Path(args.visualization_root)
        if not (vis_root / "data.yaml").exists():
            raise FileNotFoundError(
                f"Visualization root missing data.yaml: {vis_root / 'data.yaml'}"
            )

        annotated_path = run_yolo_detection(
            input_path=str(input_path),
            yolo_model_path=str(yolo_model_path),
            output_dir=args.output_dir,
            visualization_root=args.visualization_root,
        )
        print(f"YOLO output image: {annotated_path}")


if __name__ == "__main__":
    main()
