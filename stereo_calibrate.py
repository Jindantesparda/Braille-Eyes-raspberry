import cv2
import numpy as np
import time
import argparse
import platform

# ============================================================
# CONFIGURATION
# ============================================================

LEFT_CAMERA = 0
RIGHT_CAMERA = 1

RESOLUTIONS = {
    "320x240": (320, 240),
    "640x480": (640, 480),
}

parser = argparse.ArgumentParser(
    description="Capture stereo calibration images at a supported resolution."
)
parser.add_argument(
    "--resolution",
    choices=RESOLUTIONS,
    default="320x240",
    help="Capture and calibration size (default: 320x240)",
)
args = parser.parse_args()
IMAGE_WIDTH, IMAGE_HEIGHT = RESOLUTIONS[args.resolution]

# ============================================================
# CHESSBOARD CONFIGURATION
# ============================================================
# Physical chessboard:
#   8 squares wide
#   6 squares high
#
# Therefore the number of INTERNAL corners is:
#   7 x 5
#
CHESSBOARD_SIZE = (7, 5)

# Size of ONE physical square in meters.
# Change this to match your chessboard.
# Example: 25 mm = 0.025 m
SQUARE_SIZE = 0.025

# Number of successful stereo views
NUM_IMAGES = 25

OUTPUT_FILE = "stereo_calibration.npz"


# ============================================================
# PREPARE 3D CHESSBOARD POINTS
# ============================================================

objp = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE


# ============================================================
# STORAGE
# ============================================================

object_points = []
left_image_points = []
right_image_points = []


# ============================================================
# OPEN CAMERAS
# ============================================================

backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
left_cam = cv2.VideoCapture(LEFT_CAMERA, backend)

right_cam = cv2.VideoCapture(RIGHT_CAMERA, backend)

left_cam.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    IMAGE_WIDTH
)

left_cam.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    IMAGE_HEIGHT
)

right_cam.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    IMAGE_WIDTH
)

right_cam.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    IMAGE_HEIGHT
)


if not left_cam.isOpened():
    raise RuntimeError(
        "Could not open LEFT camera"
    )

if not right_cam.isOpened():
    raise RuntimeError(
        "Could not open RIGHT camera"
    )


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print("       STEREO CAMERA CALIBRATION")
print("==============================================")
print()

print(
    f"Physical chessboard: 8 x 6 squares"
)

print(
    f"Internal corners: {CHESSBOARD_SIZE}"
)

print(
    f"Square size: {SQUARE_SIZE} meters"
)

print()
print("CONTROLS")
print("SPACE = capture calibration image")
print("Q     = quit")
print()

print(
    f"Collecting {NUM_IMAGES} good stereo views."
)

print()


# ============================================================
# CAPTURE CALIBRATION IMAGES
# ============================================================

captured = 0

while captured < NUM_IMAGES:

    ret_left, left_frame = left_cam.read()
    ret_right, right_frame = right_cam.read()

    if not ret_left or not ret_right:
        print("Failed to read cameras.")
        break

    # Make sure both frames have the desired size
    left_frame = cv2.resize(
        left_frame,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    right_frame = cv2.resize(
        right_frame,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # ========================================================
    # GRAYSCALE
    # ========================================================

    left_gray = cv2.cvtColor(
        left_frame,
        cv2.COLOR_BGR2GRAY
    )

    right_gray = cv2.cvtColor(
        right_frame,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # FIND CHESSBOARD
    # ========================================================

    found_left, corners_left = cv2.findChessboardCorners(
        left_gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found_right, corners_right = cv2.findChessboardCorners(
        right_gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    display_left = left_frame.copy()
    display_right = right_frame.copy()

    # ========================================================
    # REFINE LEFT CORNERS
    # ========================================================

    if found_left:

        corners_left = cv2.cornerSubPix(
            left_gray,
            corners_left,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        cv2.drawChessboardCorners(
            display_left,
            CHESSBOARD_SIZE,
            corners_left,
            found_left
        )

    # ========================================================
    # REFINE RIGHT CORNERS
    # ========================================================

    if found_right:

        corners_right = cv2.cornerSubPix(
            right_gray,
            corners_right,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        cv2.drawChessboardCorners(
            display_right,
            CHESSBOARD_SIZE,
            corners_right,
            found_right
        )

    # ========================================================
    # STATUS
    # ========================================================

    if found_left and found_right:

        status = "CHESSBOARD FOUND - PRESS SPACE"
        status_color = (0, 255, 0)

    elif found_left:

        status = "LEFT FOUND - RIGHT NOT FOUND"
        status_color = (0, 0, 255)

    elif found_right:

        status = "RIGHT FOUND - LEFT NOT FOUND"
        status_color = (0, 0, 255)

    else:

        status = "CHESSBOARD NOT FOUND"
        status_color = (0, 0, 255)

    # ========================================================
    # LEFT CAMERA TEXT
    # ========================================================

    cv2.putText(
        display_left,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )

    cv2.putText(
        display_left,
        f"Captured: {captured}/{NUM_IMAGES}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # ========================================================
    # RIGHT CAMERA TEXT
    # ========================================================

    cv2.putText(
        display_right,
        f"Captured: {captured}/{NUM_IMAGES}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "LEFT - Calibration",
        display_left
    )

    cv2.imshow(
        "RIGHT - Calibration",
        display_right
    )

    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # CAPTURE
    # ========================================================

    if key == 32:

        if found_left and found_right:

            object_points.append(
                objp.copy()
            )

            left_image_points.append(
                corners_left
            )

            right_image_points.append(
                corners_right
            )

            captured += 1

            print(
                f"Captured stereo view "
                f"{captured}/{NUM_IMAGES}"
            )

            # Prevent accidental double captures
            time.sleep(0.3)

        else:

            print(
                "Chessboard must be visible "
                "in BOTH cameras."
            )

    # ========================================================
    # QUIT
    # ========================================================

    elif key == ord("q"):

        print("Calibration cancelled.")

        left_cam.release()
        right_cam.release()

        cv2.destroyAllWindows()

        exit()


# ============================================================
# RELEASE CAMERAS
# ============================================================

left_cam.release()
right_cam.release()

cv2.destroyAllWindows()


# ============================================================
# VALIDATE CAPTURE
# ============================================================

if captured < 10:

    raise RuntimeError(
        "Not enough calibration images. "
        "Collect at least 10-15 good views."
    )


print()
print("==============================================")
print("CAPTURE COMPLETE")
print("==============================================")
print()


# ============================================================
# IMAGE SIZE
# ============================================================

image_size = (
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)


# ============================================================
# CALIBRATE LEFT CAMERA
# ============================================================

print("Calibrating LEFT camera...")

left_rms, \
left_camera_matrix, \
left_dist_coeffs, \
_, _ = cv2.calibrateCamera(
    object_points,
    left_image_points,
    image_size,
    None,
    None
)

print(
    f"Left RMS error: "
    f"{left_rms:.4f}"
)


# ============================================================
# CALIBRATE RIGHT CAMERA
# ============================================================

print()
print("Calibrating RIGHT camera...")

right_rms, \
right_camera_matrix, \
right_dist_coeffs, \
_, _ = cv2.calibrateCamera(
    object_points,
    right_image_points,
    image_size,
    None,
    None
)

print(
    f"Right RMS error: "
    f"{right_rms:.4f}"
)


# ============================================================
# STEREO CALIBRATION
# ============================================================

print()
print("Performing stereo calibration...")

stereo_criteria = (
    cv2.TERM_CRITERIA_EPS |
    cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-5
)

# Keep the individual camera intrinsics fixed
flags = cv2.CALIB_FIX_INTRINSIC


stereo_rms, \
_, _, \
_, _, \
R, T, E, F = cv2.stereoCalibrate(

    object_points,

    left_image_points,
    right_image_points,

    left_camera_matrix,
    left_dist_coeffs,

    right_camera_matrix,
    right_dist_coeffs,

    image_size,

    criteria=stereo_criteria,
    flags=flags
)


# ============================================================
# STEREO RESULTS
# ============================================================

print()
print("==============================================")
print("STEREO CALIBRATION RESULT")
print("==============================================")
print()

print(
    f"Stereo RMS error: "
    f"{stereo_rms:.4f}"
)

print()

print("Rotation matrix R:")
print(R)

print()

print("Translation vector T:")
print(T)


# ============================================================
# CAMERA BASELINE
# ============================================================

baseline = np.linalg.norm(T)

print()

print(
    f"Camera baseline: "
    f"{baseline:.4f} meters"
)

print(
    f"Camera baseline: "
    f"{baseline * 100:.2f} cm"
)


# ============================================================
# STEREO RECTIFICATION
# ============================================================

print()
print("Computing stereo rectification...")


R1, \
R2, \
P1, \
P2, \
Q, \
roi_left, \
roi_right = cv2.stereoRectify(

    left_camera_matrix,
    left_dist_coeffs,

    right_camera_matrix,
    right_dist_coeffs,

    image_size,

    R,
    T,

    flags=cv2.CALIB_ZERO_DISPARITY,

    alpha=0
)


# ============================================================
# LEFT RECTIFICATION MAP
# ============================================================

left_map1, left_map2 = \
    cv2.initUndistortRectifyMap(

        left_camera_matrix,
        left_dist_coeffs,

        R1,
        P1,

        image_size,

        cv2.CV_32FC1
    )


# ============================================================
# RIGHT RECTIFICATION MAP
# ============================================================

right_map1, right_map2 = \
    cv2.initUndistortRectifyMap(

        right_camera_matrix,
        right_dist_coeffs,

        R2,
        P2,

        image_size,

        cv2.CV_32FC1
    )


# ============================================================
# SAVE CALIBRATION
# ============================================================

np.savez(

    OUTPUT_FILE,

    # --------------------------------------------------------
    # Camera matrices
    # --------------------------------------------------------

    left_camera_matrix=left_camera_matrix,

    right_camera_matrix=right_camera_matrix,

    # --------------------------------------------------------
    # Distortion coefficients
    # --------------------------------------------------------

    left_dist_coeffs=left_dist_coeffs,

    right_dist_coeffs=right_dist_coeffs,

    # --------------------------------------------------------
    # Stereo transformation
    # --------------------------------------------------------

    R=R,
    T=T,

    # --------------------------------------------------------
    # Essential / Fundamental matrices
    # --------------------------------------------------------

    E=E,
    F=F,

    # --------------------------------------------------------
    # Rectification
    # --------------------------------------------------------

    R1=R1,
    R2=R2,

    P1=P1,
    P2=P2,

    Q=Q,

    # --------------------------------------------------------
    # Rectification maps
    # --------------------------------------------------------

    left_map1=left_map1,
    left_map2=left_map2,

    right_map1=right_map1,
    right_map2=right_map2,

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,

    chessboard_width=CHESSBOARD_SIZE[0],
    chessboard_height=CHESSBOARD_SIZE[1],

    square_size=SQUARE_SIZE,

    baseline=baseline,

    # --------------------------------------------------------
    # Calibration errors
    # --------------------------------------------------------

    left_rms=left_rms,
    right_rms=right_rms,
    stereo_rms=stereo_rms
)


# ============================================================
# DONE
# ============================================================

print()
print("==============================================")
print("CALIBRATION SAVED")
print("==============================================")
print()

print(
    f"File: {OUTPUT_FILE}"
)

print()

print("Calibration contains:")

print("  - Left camera intrinsics")
print("  - Right camera intrinsics")
print("  - Lens distortion")
print("  - Stereo rotation")
print("  - Stereo translation")
print("  - Camera baseline")
print("  - Rectification matrices")
print("  - Q depth matrix")
print("  - Rectification maps")

print()
print("Calibration complete!")
