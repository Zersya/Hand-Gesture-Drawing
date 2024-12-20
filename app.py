from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit, join_room
import cv2
import mediapipe as mp
import numpy as np
import json
import base64
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode="threading",  
                   ping_timeout=60,
                   ping_interval=25)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,  # Adjusted confidence
    min_tracking_confidence=0.6   # Adjusted confidence
)
mp_draw = mp.solutions.drawing_utils

# Global variables for room management
rooms = {}
camera_status = {}  # Changed to store user_id: status
cameras = {}  # Track camera instances per user

def is_index_finger_up(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y

def is_middle_finger_up(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y

def is_pinky_finger_up(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

def process_hand_landmarks(hand_landmarks):
    try:
        index_tip = {
            'x': float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x),
            'y': float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
        }
        middle_tip = {
            'x': float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x),
            'y': float(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)
        }
        pinky_tip = {
            'x': float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x),
            'y': float(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y)
        }
        
        index_up = is_index_finger_up(hand_landmarks)
        middle_up = is_middle_finger_up(hand_landmarks)
        pinky_up = is_pinky_finger_up(hand_landmarks)
        
        return {'index': index_tip, 'middle': middle_tip, 'index_up': index_up, 'middle_up': middle_up, 'pinky': pinky_tip, 'pinky_up': pinky_up}
    except Exception as e:
        logger.error(f"Error processing hand landmarks: {str(e)}")
        return None

def generate_frames(room_id, user_id):
    try:
        camera = cameras.get(user_id)
        if not camera:
            logger.info(f"Attempting to open camera for user {user_id}...")
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                logger.error("Failed to open camera")
                return
            cameras[user_id] = camera  # Store camera instance
            logger.info("Camera opened successfully")
        
        last_emit_time = 0
        
        while camera_status.get(user_id, True):
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame from camera")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            processed_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    processed_hand = process_hand_landmarks(hand_landmarks)
                    if processed_hand:
                        processed_landmarks.append(processed_hand)

                current_time = time.time()
                if processed_landmarks and (current_time - last_emit_time > 0.1):  # Throttle to 10 emits per second
                    try:
                        socketio.emit('hand_position', {
                            'room': room_id,
                            'landmarks': processed_landmarks,
                            'user_id': user_id  # Add user_id to track which user's hand position
                        }, to=room_id)
                        last_emit_time = current_time
                    except Exception as e:
                        logger.error(f"Error emitting hand position: {str(e)}")

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.error("Failed to encode frame")
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Error encoding frame: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
    finally:
        if camera:
            logger.info(f"Releasing camera for user {user_id}...")
            camera.release()
            del cameras[user_id]

        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<room_id>/<user_id>')
def video_feed(room_id, user_id):
    return Response(generate_frames(room_id, user_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('join')
def on_join(data):
    room = data['room']
    print(f"Client {request.sid} joined room: {room}")  # Debug print
    
    join_room(room)

    if room not in rooms:
        rooms[room] = set()
    rooms[room].add(request.sid)
    camera_status[request.sid] = True

    emit('join_response', {'room': room, 'count': len(rooms[room])}, to=room)
    
    logger.info(f"User {request.sid} joined room {room}")

@socketio.on('leave')
def on_leave(data):
    room = data['room']
    if room in rooms:
        rooms[room].discard(request.sid)  # Use discard to avoid KeyError
        if len(rooms[room]) == 0:
            del rooms[room]
            camera_status[request.sid] = False
            # Ensure camera is released when the last user leaves
            if request.sid in cameras:
                cameras[request.sid].release()
                del cameras[request.sid]
        logger.info(f"User {request.sid} left room {room}")

@socketio.on('disconnect')
def on_disconnect():
    for room_id, participants in list(rooms.items()):
        if request.sid in participants:
            rooms[room_id].remove(request.sid)
            if len(rooms[room_id]) == 0:
                del rooms[room_id]
            if request.sid in camera_status:
                camera_status[request.sid] = False
                if request.sid in cameras:
                    cameras[request.sid].release()
                    del cameras[request.sid]
                del camera_status[request.sid]
            socketio.emit('join_response', {'count': len(rooms.get(room_id, set()))}, room=room_id)
            logger.info(f"User {request.sid} disconnected from room {room_id}")

@socketio.on('toggle_camera')
def toggle_camera(data):
    room = data.get('room')
    user_id = request.sid
    camera_status[user_id] = not camera_status.get(user_id, True)
    emit('camera_status', {'status': camera_status[user_id], 'user_id': user_id}, to=room)
    logger.info(f"Camera toggled for user {user_id}: {camera_status[user_id]}")

@socketio.on('draw')
def on_draw(data):
    emit('draw_broadcast', data, to=data['room'], include_self=False) # Changed include_self to False

@socketio.on('clear')
def on_clear(data):
    emit('clear_broadcast', data, to=data['room'])

if __name__ == '__main__':
    logger.info("Starting application...")
    socketio.run(app, 
                host='0.0.0.0', 
                port=8000,
                debug=False,  
                allow_unsafe_werkzeug=True)  