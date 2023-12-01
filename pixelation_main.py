import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection.
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create a VideoCapture object to access the webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully.
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        # Capture frame-by-frame.
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame with MediaPipe Face Detection.
        results = face_detection.process(frame_rgb)

        # Pixelate the face detection annotations on the frame.
        if results.detections:
            for detection in results.detections:
                # Get the bounding box of the face.
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Extract the face region.
                face = frame[y:y+h, x:x+w]

                # Increase the division factor for more pronounced pixelation.
                small_face = cv2.resize(face, (w // 30, h // 30), interpolation=cv2.INTER_LINEAR)
                pixelated_face = cv2.resize(small_face, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Replace the original face region with the pixelated face.
                frame[y:y+h, x:x+w] = pixelated_face

        # Display the resulting frame.
        cv2.imshow('Webcam - Pixelated Face', frame)

        # Press 'q' to exit the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture.
cap.release()
cv2.destroyAllWindows()
