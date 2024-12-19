import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Get camera resolution
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a black canvas with same resolution as camera
canvas = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
prev_point = None
drawing_color = (255, 255, 255)  # White color for drawing
drawing_thickness = 2
eraser_size = 20

def is_index_finger_up(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y

def is_middle_finger_up(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y

def draw_pointer(img, x, y, is_eraser=False):
    if is_eraser:
        # Draw eraser pointer (circle)
        cv2.circle(img, (x, y), eraser_size, (0, 0, 255), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    else:
        # Draw pen pointer (small circle)
        cv2.circle(img, (x, y), drawing_thickness + 2, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Process the image and find hands
    results = hands.process(image_rgb)

    # Draw hand landmarks and handle drawing
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Create a copy of canvas for displaying with pointer
    display_canvas = canvas.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * camera_width)
            y = int(index_finger_tip.y * camera_height)
            
            # Check if index finger is up
            index_up = is_index_finger_up(hand_landmarks)
            middle_up = is_middle_finger_up(hand_landmarks)
            
            if index_up:
                if middle_up:  # Eraser mode (both fingers up)
                    draw_pointer(display_canvas, x, y, is_eraser=True)
                    draw_pointer(image, x, y, is_eraser=True)
                    if prev_point is not None:
                        cv2.circle(canvas, (x, y), eraser_size, (0, 0, 0), -1)
                    prev_point = (x, y)
                elif not middle_up:  # Drawing mode (only index finger up)
                    draw_pointer(display_canvas, x, y, is_eraser=False)
                    draw_pointer(image, x, y, is_eraser=False)
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (x, y), drawing_color, drawing_thickness)
                    prev_point = (x, y)
            else:
                prev_point = None

    # Display the image and canvas
    cv2.imshow('Hand Gesture Recognition', image)
    cv2.imshow('Drawing Canvas', display_canvas)
    
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
    elif key == ord('+'):  # Increase eraser size
        eraser_size = min(100, eraser_size + 5)
    elif key == ord('-'):  # Decrease eraser size
        eraser_size = max(5, eraser_size - 5)

# Release resources
cap.release()
cv2.destroyAllWindows()
