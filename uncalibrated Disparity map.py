import cv2
import numpy as np

# ============================================================
# CAMERA SETTINGS
# ============================================================

LEFT_CAMERA = 0
RIGHT_CAMERA = 1

WIDTH = 640
HEIGHT = 480

# ============================================================
# OPEN CAMERAS
# ============================================================

left_cam = cv2.VideoCapture(LEFT_CAMERA, cv2.CAP_DSHOW)
right_cam = cv2.VideoCapture(RIGHT_CAMERA, cv2.CAP_DSHOW)

left_cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
left_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

right_cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
right_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not left_cam.isOpened():
    print("Could not open left camera")
    exit()

if not right_cam.isOpened():
    print("Could not open right camera")
    exit()

# ============================================================
# STEREO SGBM
# ============================================================

# Number of disparities must be divisible by 16
num_disparities = 128

block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disparities,
    blockSize=block_size,

    P1=8 * 1 * block_size ** 2,
    P2=32 * 1 * block_size ** 2,

    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,

    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# ============================================================
# MAIN LOOP
# ============================================================

print("Stereo vision running.")
print("Press Q to quit.")

while True:

    ret_left, left_frame = left_cam.read()
    ret_right, right_frame = right_cam.read()

    if not ret_left or not ret_right:
        print("Failed to read cameras")
        break

    # Make sure both images are the same size
    left_frame = cv2.resize(left_frame, (WIDTH, HEIGHT))
    right_frame = cv2.resize(right_frame, (WIDTH, HEIGHT))

    # Convert to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # COMPUTE DISPARITY
    # ========================================================

    disparity = stereo.compute(left_gray, right_gray)

    # Convert from fixed point representation
    disparity = disparity.astype(np.float32) / 16.0

    # ========================================================
    # NORMALIZE FOR DISPLAY
    # ========================================================

    disparity_display = cv2.normalize(
        disparity,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    disparity_display = disparity_display.astype(np.uint8)

    # Apply color map
    disparity_color = cv2.applyColorMap(
        disparity_display,
        cv2.COLORMAP_JET
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow("Left Camera", left_frame)
    cv2.imshow("Right Camera", right_frame)
    cv2.imshow("Disparity Map", disparity_color)

    # Quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()