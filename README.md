# Webcam Eye Blocker

The Webcam Eye Blocker is a Python application that uses real-time face detection to draw a black rectangle over the eyes. It is built with OpenCV for image processing and the MediaPipe library for accurate face landmark detection.

## Sample Output

Here is an example of what the Webcam Eye Blocker output looks like:

## Limitations

The application is designed to work in real-time with a standard webcam feed. As such, the performance and accuracy are subject to lighting conditions, webcam quality, and the individual's positioning in front of the camera. The rectangle drawn over the eyes is a simple overlay and does not adapt dynamically to head movements beyond the scope of MediaPipe's face detection model.

## Setup and Run

To set up and run the Webcam Eye Blocker, install the required dependencies (Mediapipe and OpenCV), and run the python file.
