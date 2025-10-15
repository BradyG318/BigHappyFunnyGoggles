import cv2
import mediapipe as mp

# Function to preprocess the frame for better face detection
def preprocess_frame(image):
    # Reduce compression artifacts
    image = cv2.medianBlur(image, 5)  # Reduce noise aggressively for longer range
    
    # Enhance contrast aggressively for longer range (helps with detection)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0).apply(lab[:,:,0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Scale image up for better detection of smaller faces
    scale_factor = 1.5  # Increase this if needed (1.5 = 150% size)
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

    return image

# Initialize MediaPipe variables
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0) # change number to use different cameras

# Use LONG RANGE face detection model
with mp_face_detection.FaceDetection(
    model_selection=1,  # 1 = Long range model
    min_detection_confidence=0.1
) as face_detection:

  # Use MediaPipe Face Detection
  with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=False,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.1) as face_mesh:

    while cap.isOpened(): # While the camera is running
      success, image = cap.read() # If camera running is succesful, read the image
      
      if not success: # If not successful, print error and continue
        print("Ignoring empty camera frame.")
        continue

      # Preprocess the frame
      image = preprocess_frame(image)
      image.flags.writeable = False # To improve performance, mark the image as not writeable
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB for MediaPipe

      # Process with Face Detection first (better for long range)
      detection_results = face_detection.process(image_rgb)

      if detection_results.detections:
        for detection in detection_results.detections:
          # Get bounding box of detected face
          bbox = detection.location_data.relative_bounding_box
          h, w, _ = image.shape

          x_min = int(bbox.xmin * w)
          y_min = int(bbox.ymin * h)
          box_w = int(bbox.width * w)
          box_h = int(bbox.height * h)

          # Expand ROI slightly for better mesh detection
          expansion = 0.15  # 15% expansion
          x_min = max(0, int(x_min - box_w * expansion))
          y_min = max(0, int(y_min - box_h * expansion))
          x_max = min(w, int(x_min + box_w * (1 + 2 * expansion)))
          y_max = min(h, int(y_min + box_h * (1 + 2 * expansion)))

          # Extract face region
          face_roi = image[y_min:y_max, x_min:x_max]

          if face_roi.size > 0:
            # Process ROI with Face Mesh
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(face_roi_rgb)
                        
            if mesh_results.multi_face_landmarks:
              # Draw bounding box (from face detection)
              image.flags.writeable = True
              cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

      # Create a named window to display the image in fullscreen mode
      cv2.namedWindow('MediaPipe Hybrid Face Detection - Long Range', cv2.WND_PROP_FULLSCREEN)
      cv2.setWindowProperty('MediaPipe Hybrid Face Detection - Long Range', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
      
      # To flip the image horizontally for a selfie-view display, replace image --> cv2.flip(image, 1)
      # Display the image with detected faces in the named window
      cv2.imshow('MediaPipe Hybrid Face Detection - Long Range', image)
      
      # Break the loop on 'ESC' key press
      if cv2.waitKey(5) & 0xFF == 27:
        break
  
# Release the camera and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()