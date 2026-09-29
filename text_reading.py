"""User-triggered high-resolution OCR mode; OCR is never run in navigation mode."""

import argparse
import platform
import shutil
import subprocess

import cv2


RESOLUTIONS = {
    "320x240": (320, 240),
    "640x480": (640, 480),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture a still, select its text area, and read it with OCR."
    )
    parser.add_argument("--resolution", choices=RESOLUTIONS, default="640x480")
    parser.add_argument("--camera-left", type=int, default=0)
    parser.add_argument("--camera-right", type=int, default=1)
    parser.add_argument(
        "--tesseract-cmd", default=None,
        help="Optional path to the Tesseract executable",
    )
    parser.add_argument(
        "--no-speech", action="store_true",
        help="Print OCR results without speaking them through espeak-ng/espeak",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import pytesseract
    except ImportError as error:
        raise SystemExit(
            "OCR needs pytesseract. Install project requirements and Tesseract OCR."
        ) from error

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    width, height = RESOLUTIONS[args.resolution]
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
    cameras = [
        cv2.VideoCapture(args.camera_left, backend),
        cv2.VideoCapture(args.camera_right, backend),
    ]
    for camera in cameras:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, 5)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not all(camera.isOpened() for camera in cameras):
        for camera in cameras:
            camera.release()
        raise RuntimeError("Could not open both cameras for text-reading mode")

    print("Text-reading mode: SPACE captures/selects text; B or Q returns to navigation.")
    try:
        while True:
            ok_left, left_frame = cameras[0].read()
            ok_right, right_frame = cameras[1].read()
            if not ok_left or not ok_right:
                print("Camera frame read failed.")
                break
            if left_frame.shape[1::-1] != (width, height):
                actual_width, actual_height = left_frame.shape[1::-1]
                print(
                    f"Left camera returned {actual_width}x{actual_height}; "
                    "text capture uses its native resolution."
                )
                width, height = actual_width, actual_height

            cv2.putText(
                left_frame, "SPACE: capture and select text | B/Q: navigation",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                cv2.LINE_AA,
            )
            cv2.imshow("Text mode - left camera", left_frame)
            cv2.imshow("Text mode - right camera", right_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("b"), ord("q")):
                break
            if key != ord(" "):
                continue

            # Freeze one high-resolution left image and let the user crop the text.
            capture = left_frame.copy()
            x, y, crop_width, crop_height = cv2.selectROI(
                "Select text area (ENTER to OCR)", capture,
                showCrosshair=True, fromCenter=False,
            )
            if crop_width == 0 or crop_height == 0:
                crop_width = int(width * 0.8)
                crop_height = int(height * 0.8)
                x = (width - crop_width) // 2
                y = (height - crop_height) // 2
            crop = capture[y:y + crop_height, x:x + crop_width]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            enlarged = cv2.resize(
                gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )
            prepared = cv2.threshold(
                enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
            try:
                text = pytesseract.image_to_string(
                    prepared, config="--oem 1 --psm 6"
                ).strip()
            except pytesseract.TesseractNotFoundError as error:
                print("Tesseract executable not found. Install it or set --tesseract-cmd.")
                print(error)
                continue
            except pytesseract.TesseractError as error:
                print("Tesseract failed to process the selected crop:")
                print(error)
                continue

            print("\n--- OCR result ---")
            print(text if text else "No text recognized.")
            print("------------------\n")
            if text and not args.no_speech:
                speech_engine = shutil.which("espeak-ng") or shutil.which("espeak")
                if speech_engine:
                    subprocess.run(
                        [speech_engine, "-s", "150", text], check=False
                    )
            cv2.imshow("OCR text crop", crop)
            cv2.waitKey(1)
    finally:
        for camera in cameras:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
