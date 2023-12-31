# Webcam Eye Blocker

The Webcam Eye Blocker is a Python application that uses real-time face detection to draw a black rectangle over the eyes. There's also another piece of code that does pixelation. It is built with OpenCV for image processing and the MediaPipe library for accurate face landmark detection.

## Sample Output

Here is an example of what the Webcam Eye Blocker output looks like:

![demo](https://github.com/johngunerli/webcam-eye-blocker/assets/33205097/d707b8eb-6d1c-4180-a8b3-ec2155efa20a)

Here's the pixelation code output: 

<img width="1033" alt="Screenshot 2023-11-30 at 7 10 01 PM" src="https://github.com/johngunerli/webcam-eye-blocker/assets/33205097/78fd4d8b-82cd-47b8-876e-8f9fdf3b7499">



## Limitations

The application is designed to work in real-time with a standard webcam feed. As such, the performance and accuracy are subject to lighting conditions, webcam quality, and the individual's positioning in front of the camera. The rectangle drawn over the eyes is a simple overlay and does not adapt dynamically to head movements beyond the scope of MediaPipe's face detection model.

## Setup and Run

To set up and run the Webcam Eye Blocker, install the required dependencies (Mediapipe and OpenCV), and run the python file.
