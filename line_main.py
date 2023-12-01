import cv2
import mediapipe as mp

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

        # Draw the face detection annotations on the frame.
        if results.detections:
            for detection in results.detections:
                # Get the bounding box and keypoints of the face.
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]

                # Calculate the coordinates on the frame.
                re_x, re_y = int(right_eye.x * iw), int(right_eye.y * ih)
                le_x, le_y = int(left_eye.x * iw), int(left_eye.y * ih)

                # Extend the rectangle to be 3 times the distance between the eyes
                eye_distance = re_x - le_x
                start_point = (max(le_x - eye_distance, 0), le_y - 10)  # 10 pixels above the eye
                end_point = (min(re_x + eye_distance, iw), re_y + 10)  # 10 pixels below the eye

                # Increase the thickness of the rectangle.
                thickness = -1  # -1 will fill the rectangle

                # Draw a black rectangle across the eyes.
                cv2.rectangle(frame, start_point, end_point, (0, 0, 0), thickness)

        # Display the resulting frame.
        cv2.imshow('Webcam - Eye Detection', frame)

        # Press 'q' to exit the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture.
cap.release()
cv2.destroyAllWindows()
