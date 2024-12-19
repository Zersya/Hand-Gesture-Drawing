import cv2
import mediapipe as mp
import numpy as np
import websockets
import asyncio
import json
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# WebSocket configuration
WEBSOCKET_URI = "ws://localhost:8000/ws/room1"
websocket = None
canvas_lock = threading.Lock()

# Camera and canvas settings
camera_enabled = True
mouse_drawing = False
mouse_position = None
drawing_mode = True  # True for draw, False for erase

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

async def websocket_client():
    global websocket
    while True:
        try:
            async with websockets.connect(WEBSOCKET_URI) as ws:
                websocket = ws
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    if data["type"] == "draw":
                        with canvas_lock:
                            cv2.line(canvas, 
                                   (data["start_x"], data["start_y"]), 
                                   (data["end_x"], data["end_y"]), 
                                   tuple(data["color"]), 
                                   data["thickness"])
        except:
            await asyncio.sleep(1)

def start_websocket_client():
    asyncio.run(websocket_client())

# Start WebSocket client in a separate thread
websocket_thread = threading.Thread(target=start_websocket_client, daemon=True)
websocket_thread.start()

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

async def send_draw_data(start_point, end_point, color, thickness):
    if websocket:
        try:
            data = {
                "type": "draw",
                "start_x": int(start_point[0]),
                "start_y": int(start_point[1]),
                "end_x": int(end_point[0]),
                "end_y": int(end_point[1]),
                "color": color,
                "thickness": thickness
            }
            await websocket.send(json.dumps(data))
        except:
            pass

def send_draw_data_sync(start_point, end_point, color, thickness):
    asyncio.run(send_draw_data(start_point, end_point, color, thickness))

def mouse_callback(event, x, y, flags, param):
    global mouse_position, mouse_drawing, prev_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_drawing = True
        mouse_position = (x, y)
        prev_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and mouse_drawing:
        if prev_point is not None:
            if drawing_mode:
                cv2.line(canvas, prev_point, (x, y), drawing_color, drawing_thickness)
                send_draw_data_sync(prev_point, (x, y), drawing_color, drawing_thickness)
            else:
                cv2.circle(canvas, (x, y), eraser_size, (0, 0, 0), -1)
                send_draw_data_sync(prev_point, (x, y), (0, 0, 0), eraser_size)
        prev_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_drawing = False
        prev_point = None

# Create window and set mouse callback
cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas', mouse_callback)

while True:
    display_canvas = canvas.copy()
    
    if camera_enabled and cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get index finger tip coordinates
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * camera_width)
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * camera_height)
                
                # Check fingers state
                index_up = is_index_finger_up(hand_landmarks)
                middle_up = is_middle_finger_up(hand_landmarks)
                
                if index_up:
                    if middle_up:  # Eraser mode
                        draw_pointer(image, x, y, is_eraser=True)
                        if prev_point is not None:
                            cv2.circle(canvas, (x, y), eraser_size, (0, 0, 0), -1)
                            send_draw_data_sync(prev_point, (x, y), (0, 0, 0), eraser_size)
                        prev_point = (x, y)
                    elif not middle_up:  # Drawing mode
                        draw_pointer(display_canvas, x, y, is_eraser=False)
                        draw_pointer(image, x, y, is_eraser=False)
                        if prev_point is not None:
                            cv2.line(canvas, prev_point, (x, y), drawing_color, drawing_thickness)
                            send_draw_data_sync(prev_point, (x, y), drawing_color, drawing_thickness)
                        prev_point = (x, y)
                else:
                    prev_point = None
        
        # Show the camera feed in a separate window
        cv2.imshow('Camera Feed', image)
    
    # Show the canvas
    cv2.imshow('Canvas', display_canvas)
    
    # Handle keyboard input
    key = cv2.pollKey()
    if key != -1:
        key = key & 0xFF
        print(key)
        if key == ord('q'):
            break
        elif key == ord('c'):
            with canvas_lock:
                canvas = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        elif key == ord('v'):
            camera_enabled = not camera_enabled
            if not camera_enabled and cap.isOpened():
                cv2.destroyWindow('Camera Feed')
        elif key == ord('m'):
            drawing_mode = not drawing_mode
        elif key == ord('w'):
            drawing_color = (255, 255, 255)  # White
        elif key == ord('r'):
            drawing_color = (0, 0, 255)      # Red
        elif key == ord('g'):
            drawing_color = (0, 255, 0)      # Green
        elif key == ord('b'):
            drawing_color = (255, 0, 0)      # Blue

# Clean up
hands.close()
cap.release()
cv2.destroyAllWindows()
