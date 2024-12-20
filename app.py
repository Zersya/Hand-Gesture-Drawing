from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room
import logging

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

# Global variables for room management
rooms = {}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('join')
def on_join(data):
    room = data['room']
    logger.info(f"Client {request.sid} joined room: {room}")
    
    join_room(room)

    if room not in rooms:
        rooms[room] = set()
    rooms[room].add(request.sid)

    emit('join_response', {'room': room, 'count': len(rooms[room])}, to=room)


@socketio.on('leave')
def on_leave(data):
    room = data['room']
    if room in rooms:
        rooms[room].discard(request.sid)
        if len(rooms[room]) == 0:
            del rooms[room]
        logger.info(f"User {request.sid} left room {room}")
        emit('join_response', {'room': room, 'count': len(rooms.get(room, set()))}, to=room)

@socketio.on('disconnect')
def on_disconnect():
    for room_id, participants in list(rooms.items()):
        if request.sid in participants:
            rooms[room_id].remove(request.sid)
            if len(rooms[room_id]) == 0:
                del rooms[room_id]
            socketio.emit('join_response', {'count': len(rooms.get(room_id, set()))}, room=room_id)
            logger.info(f"User {request.sid} disconnected from room {room_id}")

@socketio.on('hand_position')
def on_hand_position(data):
    logger.debug(f"Broadcasting hand position for user {request.sid} in room {data['room']}")
    emit('hand_position_broadcast', data, to=data['room'], include_self=False)

@socketio.on('draw')
def on_draw(data):
    emit('draw_broadcast', data, to=data['room'], include_self=False)

@socketio.on('clear')
def on_clear(data):
    emit('clear_broadcast', data, to=data['room'])

if __name__ == '__main__':
    logger.info("Starting application...")
    socketio.run(app, 
                host='0.0.0.0', 
                port=8000,
                debug=True,  
                allow_unsafe_werkzeug=True)
