import cv2
import mediapipe as mp

# Initialize MediaPipe variables
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(3) # 0 is webcam, 3 is OBS virtual cam (for me)

# Use MediaPipe Face Detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4) as face_detection:
  while cap.isOpened(): # While the camera is running
    success, image = cap.read() # If camera running is succesful, read the image
    
    if not success: # If not successful, print error and continue
      print("Ignoring empty camera frame.")
      continue
    
    # Processing
    image.flags.writeable = False # To improve performance, mark the image as not writeable
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB for MediaPipe
    results = face_detection.process(image) # Process the image and detect faces

    # Draw the face detection annotations on the image.
    image.flags.writeable = True # Set the image back to writeable to be able to draw on it
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert the image back to BGR for OpenCV

    if results.detections: # Check if any faces are detected
      for detection in results.detections: # For each detected face
        mp_drawing.draw_detection(image, detection) # Draw the detection on the image

    # Flip the image horizontally for a selfie-view display (since webcam)
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    
    # Break the loop on 'ESC' key press
    if cv2.waitKey(5) & 0xFF == 27:
      break
  
# Release the camera and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()