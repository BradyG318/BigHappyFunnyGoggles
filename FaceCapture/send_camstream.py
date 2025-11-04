# simple_stream.py
import cv2
from flask import Flask, Response
from pyngrok import ngrok

# initialize Flask app reference
app = Flask(__name__)

# initialize video capture object
cap = cv2.VideoCapture(0) # 0 for default camera

# Generator function to yield camera frames
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask route to serve video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route for the main page with video feed
@app.route('/')
def index():
    return """
    <html>
    <head><title>Face Detection Stream</title></head>
    <body>
        <h1>Face Detection Live Stream</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

# Run the Flask app
if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)

    print("Starting MJPEG Stream")
    print("=" * 50)
    print(f"Ngrok URL: {public_url}")
    print(f"Local URL: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000)