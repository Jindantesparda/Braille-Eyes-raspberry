import argparse
import os
import platform
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


LEFT_CAMERA = 0
RIGHT_CAMERA = 1
RESOLUTIONS = {
    "320x240": (320, 240),
    "640x480": (640, 480),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stereo disparity with bounded, low-rate ROI processing."
    )
    parser.add_argument(
        "--resolution", choices=RESOLUTIONS, default="320x240",
        help="Capture size (default: 320x240)",
    )
    parser.add_argument(
        "--calibration", default="stereo_calibration.npz",
        help="Calibration file used to rectify frames and bound disparity by distance",
    )
    parser.add_argument(
        "--min-distance", type=float, default=0.5,
        help="Nearest useful distance in metres when calibrated (default: 0.5)",
    )
    parser.add_argument(
        "--max-distance", type=float, default=5.0,
        help="Farthest useful distance in metres when calibrated (default: 5.0)",
    )
    parser.add_argument(
        "--max-disparity", type=int, default=None,
        help="Search-range cap in pixels when no calibration file is available",
    )
    parser.add_argument(
        "--roi-fraction", type=float, default=0.5,
        help="Width and height fraction of the centered processing ROI (default: 0.5)",
    )
    parser.add_argument(
        "--camera-fps", type=float, default=30.0,
        help="Requested camera capture rate (default: 30)",
    )
    parser.add_argument(
        "--processing-fps", type=float, default=10.0,
        help="Maximum stereo depth processing rate (default: 10)",
    )
    parser.add_argument(
        "--audio", action="store_true",
        help="Enable precomputed, distance-scaled directional audio cues",
    )
    parser.add_argument(
        "--ocr-resolution", choices=RESOLUTIONS, default="640x480",
        help="Resolution used in text-reading mode (default: 640x480)",
    )
    return parser.parse_args()


def load_calibration(path, image_size):
    if not os.path.isfile(path):
        return None

    with np.load(path) as data:
        calibrated_size = (
            int(data["image_width"]),
            int(data["image_height"]),
        )
        if calibrated_size != image_size:
            raise ValueError(
                f"Calibration is for {calibrated_size[0]}x{calibrated_size[1]}, "
                f"but --resolution requests {image_size[0]}x{image_size[1]}. "
                "Run stereo_calibrate.py at the selected resolution first."
            )

        # Copy maps out of the npz archive once; each frame can then be remapped.
        maps = {
            "left_map1": data["left_map1"],
            "left_map2": data["left_map2"],
            "right_map1": data["right_map1"],
            "right_map2": data["right_map2"],
        }
        focal_px = float(data["P1"][0, 0])
        baseline_m = float(data["baseline"])

    if focal_px <= 0 or baseline_m <= 0:
        raise ValueError("Calibration must contain positive focal length and baseline")
    return maps, focal_px, baseline_m


def capture_latest(camera, name, state, state_lock, stop_event):
    sequence = 0
    while not stop_event.is_set():
        ok, frame = camera.read()
        if not ok:
            if not stop_event.is_set():
                time.sleep(0.01)
            continue
        sequence += 1
        # Keep only a reference to the latest frame; do not queue/copy old frames.
        with state_lock:
            state[name] = (sequence, time.perf_counter(), frame)


def nearest_obstacle(disparity, min_distance, max_distance, focal_px, baseline_m):
    """Estimate close surfaces by sector, ignoring isolated disparity speckles."""
    disparity_px = disparity.astype(np.float32) / 16.0
    far_px = focal_px * baseline_m / max_distance
    near_px = focal_px * baseline_m / min_distance
    candidates = []

    for direction, sector in zip(
        ("left", "center", "right"), np.array_split(disparity_px, 3, axis=1)
    ):
        valid = sector[
            (sector > 0)
            & (sector >= far_px - 0.5)
            & (sector <= near_px + 0.5)
        ]
        if valid.size < max(12, int(sector.size * 0.01)):
            continue
        near_percentile = float(np.percentile(valid, 90))
        distance = focal_px * baseline_m / near_percentile
        candidates.append((distance, direction))

    return min(candidates) if candidates else None


def main():
    args = parse_args()
    text_reader = os.path.join(os.path.dirname(__file__), "text_reading.py")
    while run_navigation(args) == "text-reading":
        subprocess.run(
            [sys.executable, text_reader, "--resolution", args.ocr_resolution],
            check=False,
        )


def run_navigation(args):
    width, height = RESOLUTIONS[args.resolution]
    if not 0.2 <= args.roi_fraction <= 1.0:
        raise ValueError("--roi-fraction must be between 0.2 and 1.0")
    if args.camera_fps <= 0 or args.processing_fps <= 0:
        raise ValueError("FPS values must be positive")
    if args.min_distance <= 0 or args.max_distance <= args.min_distance:
        raise ValueError("Distances must satisfy 0 < min-distance < max-distance")

    calibration = load_calibration(args.calibration, (width, height))
    min_disparity = 0
    if calibration:
        maps, focal_px, baseline_m = calibration
        far_disparity = focal_px * baseline_m / args.max_distance
        near_disparity = focal_px * baseline_m / args.min_distance
        min_disparity = max(0, int(np.floor(far_disparity)))
        required = max(1, int(np.ceil(near_disparity)) - min_disparity + 1)
        requested_disparities = ((required + 15) // 16) * 16
        range_source = (
            f"{args.min_distance:g}-{args.max_distance:g} m "
            f"(f={focal_px:.1f}px, baseline={baseline_m:.3f}m)"
        )
    else:
        maps = None
        requested_disparities = args.max_disparity or (64 if width == 320 else 128)
        min_disparity = 0
        range_source = "pixel cap only; no metric distance bound without calibration"
        print(
            f"Calibration not found at {args.calibration!r}; using unrectified "
            "frames. Run stereo_calibrate.py at this resolution for metric range."
        )

    # OpenCV requires the disparity count to be a positive multiple of 16.
    num_disparities = max(16, ((requested_disparities + 15) // 16) * 16)
    if args.max_disparity is not None and not calibration:
        if args.max_disparity <= 0:
            raise ValueError("--max-disparity must be positive")
        num_disparities = max(16, ((args.max_disparity + 15) // 16) * 16)
    max_supported = (width // 16) * 16
    if num_disparities > max_supported:
        raise ValueError(
            f"The requested range needs {num_disparities} disparities, but "
            f"{width}x{height} supports at most {max_supported}. Use a closer "
            "minimum distance or higher resolution."
        )

    roi_width = max(16, int(width * args.roi_fraction))
    roi_height = max(16, int(height * args.roi_fraction))
    roi_x = (width - roi_width) // 2
    roi_y = (height - roi_height) // 2
    # Include the pixels to the left needed to match objects at the ROI's left edge.
    match_x = max(0, roi_x - min_disparity - num_disparities)
    match_width = roi_x + roi_width - match_x
    if match_width <= num_disparities:
        raise ValueError(
            "The selected ROI is too narrow for this disparity range. "
            "Increase --roi-fraction, reduce the distance range, or use "
            "640x480."
        )

    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
    left_cam = cv2.VideoCapture(LEFT_CAMERA, backend)
    right_cam = cv2.VideoCapture(RIGHT_CAMERA, backend)
    cameras = (left_cam, right_cam)
    for camera in cameras:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not left_cam.isOpened() or not right_cam.isOpened():
        for camera in cameras:
            camera.release()
        raise RuntimeError("Could not open both stereo cameras")

    audio_player = None
    if args.audio:
        if calibration:
            try:
                from audio_cues import AudioCuePlayer

                audio_player = AudioCuePlayer()
            except (ImportError, RuntimeError) as error:
                print(f"Audio cues unavailable: {error}")
        else:
            print("Audio cues need a matching calibration file; audio is disabled.")

    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    state = {"left": None, "right": None}
    state_lock = threading.Lock()
    stop_event = threading.Event()
    capture_threads = [
        threading.Thread(
            target=capture_latest,
            args=(left_cam, "left", state, state_lock, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=capture_latest,
            args=(right_cam, "right", state, state_lock, stop_event),
            daemon=True,
        ),
    ]
    for thread in capture_threads:
        thread.start()

    print(f"Capture: {width}x{height} at requested {args.camera_fps:g} FPS")
    print(
        f"Stereo: up to {args.processing_fps:g} FPS, centered "
        f"{roi_width}x{roi_height} ROI"
    )
    print(
        f"Disparity search: {min_disparity}.."
        f"{min_disparity + num_disparities - 1} px ({range_source})"
    )
    print("Press T for text-reading mode, or Q to quit.")

    period = 1.0 / args.processing_fps
    next_process = 0.0
    last_sequences = (-1, -1)
    processed_frames = 0
    report_at = time.perf_counter() + 1.0
    action = "quit"
    last_audio_key = None
    last_audio_at = 0.0

    try:
        while True:
            now = time.perf_counter()
            if now >= next_process:
                with state_lock:
                    left_item = state["left"]
                    right_item = state["right"]

                if (
                    left_item is not None
                    and right_item is not None
                    and (left_item[0], right_item[0]) != last_sequences
                ):
                    left_frame = left_item[2]
                    right_frame = right_item[2]
                    if left_frame.shape[1::-1] != (width, height):
                        left_frame = cv2.resize(left_frame, (width, height))
                    if right_frame.shape[1::-1] != (width, height):
                        right_frame = cv2.resize(right_frame, (width, height))

                    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                    if maps:
                        left_gray = cv2.remap(
                            left_gray, maps["left_map1"], maps["left_map2"],
                            cv2.INTER_LINEAR,
                        )
                        right_gray = cv2.remap(
                            right_gray, maps["right_map1"], maps["right_map2"],
                            cv2.INTER_LINEAR,
                        )

                    match_y = roi_y
                    match_height = roi_height
                    left_roi = left_gray[
                        match_y:match_y + match_height,
                        match_x:match_x + match_width,
                    ]
                    right_roi = right_gray[
                        match_y:match_y + match_height,
                        match_x:match_x + match_width,
                    ]
                    disparity = stereo.compute(left_roi, right_roi)

                    # Drop the horizontal halo, keeping only the requested center ROI.
                    roi_offset = roi_x - match_x
                    disparity = disparity[
                        :, roi_offset:roi_offset + roi_width
                    ]
                    disparity_display = cv2.normalize(
                        disparity, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U,
                    )
                    disparity_color = cv2.applyColorMap(
                        disparity_display, cv2.COLORMAP_JET
                    )

                    left_display = left_gray[
                        roi_y:roi_y + roi_height, roi_x:roi_x + roi_width
                    ]
                    right_display = right_gray[
                        roi_y:roi_y + roi_height, roi_x:roi_x + roi_width
                    ]
                    cv2.imshow("Left center ROI", left_display)
                    cv2.imshow("Right center ROI", right_display)
                    cv2.imshow("Stereo disparity ROI", disparity_color)
                    if audio_player:
                        estimate = nearest_obstacle(
                            disparity, args.min_distance, args.max_distance,
                            focal_px, baseline_m,
                        )
                        if estimate is not None:
                            distance, direction = estimate
                            cue_key = (round(distance, 1), direction)
                            cue_now = time.perf_counter()
                            if (
                                cue_now - last_audio_at >= 0.7
                                or (
                                    cue_key != last_audio_key
                                    and cue_now - last_audio_at >= 0.25
                                )
                            ):
                                audio_player.play(distance, direction)
                                last_audio_key = cue_key
                                last_audio_at = cue_now
                    last_sequences = (left_item[0], right_item[0])
                    processed_frames += 1
                    next_process = time.perf_counter() + period
                else:
                    next_process = now + 0.005

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                action = "text-reading"
                break

            now = time.perf_counter()
            if now >= report_at:
                elapsed = now - (report_at - 1.0)
                print(f"Depth processing: {processed_frames / elapsed:.1f} FPS")
                processed_frames = 0
                report_at = now + 1.0
            if next_process > now:
                time.sleep(min(0.005, next_process - now))
    finally:
        stop_event.set()
        for thread in capture_threads:
            thread.join(timeout=0.5)
        for camera in cameras:
            camera.release()
        for thread in capture_threads:
            thread.join(timeout=0.5)
        if audio_player:
            audio_player.close()
        cv2.destroyAllWindows()

    return action


if __name__ == "__main__":
    main()
