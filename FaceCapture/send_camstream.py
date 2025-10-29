# simple_stream.py
import cv2
from flask import Flask, Response

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
        <p>Stream is active! View the video feed above.</p>
    </body>
    </html>
    """

# Run the Flask app
if __name__ == '__main__':
    print("""Connect at: http://10.115.106.201:5000 for website
    or http://10.115.106.201:5000/video_feed for direct feed""")
    app.run(host='10.115.106.201', port=5000) # change host IP address as needed