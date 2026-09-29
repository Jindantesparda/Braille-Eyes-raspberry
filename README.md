# Braille-Eyes Raspberry Pi stereo tools

The navigation viewer is configured for a Raspberry Pi with limited memory. It
defaults to 320×240, requests 30 FPS from the cameras, processes stereo at up to
10 FPS, and calculates disparity in a centered ROI. Capture threads keep only
each camera's newest frame, so slow depth processing does not build a queue of
stale frames.

## Calibrate and run navigation

Calibrate at the resolution you plan to use. Calibration maps are tied to that
resolution; the viewer reports an error if they do not match.

```sh
python3 stereo_calibrate.py --resolution 320x240
python3 "uncalibrated Disparity map.py" --resolution 320x240
```

The viewer loads `stereo_calibration.npz` when it exists. It uses the calibrated
focal length and camera baseline to limit disparity to the default 0.5–5 metre
range, and rectifies frames before matching. Without calibration, the viewer
uses a pixel-bounded unrectified preview; that mode cannot provide metric depth
or audio distance cues.

Press **T** in the navigation window to switch to text-reading mode. Navigation
releases the cameras first; the text mode then requests a higher-resolution
capture. Press **Space**, select the text region with the mouse, and press
Enter. OCR runs only on that captured crop. The recognized text is printed to
the terminal. Press **B** or **Q** to return to navigation.

To compare 640×480, calibrate and run navigation at that resolution:

```sh
python3 stereo_calibrate.py --resolution 640x480
python3 "uncalibrated Disparity map.py" --resolution 640x480
```

## Audio cues

Install the Python requirements, enable stereo audio output, and start the
viewer with `--audio`:

```sh
python3 -m pip install -r requirements.txt
python3 "uncalibrated Disparity map.py" --resolution 320x240 --audio
```

On Raspberry Pi OS, the Tesseract OCR engine is a system package in addition to
the Python `pytesseract` wrapper:

```sh
sudo apt install tesseract-ocr espeak-ng
```

When `espeak-ng` (or `espeak`) is installed, OCR results are also spoken. Pass
`--no-speech` to `text_reading.py` to keep the result in the terminal only.

The viewer precomputes short 22.05 kHz cues at startup. With calibration, it
uses the 90th-percentile valid disparity in left, center, and right sections of
the ROI to select a panned direction and a distance band. Nearer obstacles use
higher-pitched, louder cues; cues are rate-limited to at most four changes per
second and repeat while the obstacle persists. This is a basic disparity
heuristic for feedback, not a validated obstacle classifier or room-reverb
simulation. Keep `--audio` off to avoid initializing the sound device.

Useful options include:

```text
--resolution 320x240|640x480
--min-distance 0.5       Nearest useful range in metres (calibrated only)
--max-distance 5.0        Farthest useful range in metres (calibrated only)
--roi-fraction 0.5        Center ROI width and height as frame fractions
--camera-fps 30           Requested camera capture rate
--processing-fps 10       Maximum stereo processing rate
--ocr-resolution 640x480   Text-reading capture resolution
--audio                   Enable calibrated audio cues
```

OCR requires the Tesseract executable to be installed and available on PATH. If
it is elsewhere, run `text_reading.py` directly with `--tesseract-cmd PATH`.
Verify obstacle reliability, OCR readability, and audio cues on the target
cameras and Raspberry Pi before relying on them for navigation.
