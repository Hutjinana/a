from flask import Flask, redirect, url_for, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

camera_running = False
cap = None

def generate_frames():
    global cap, camera_running
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    camera_running = True
    while camera_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if cap:
        cap.release()
    cv2.destroyAllWindows()

def generate_frames2():
    global cap, camera_running
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    camera_running = True
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))             

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if cap:
        cap.release()
    cv2.destroyAllWindows()

@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/one")
def cam1():
    return render_template("cam1.html")

@app.route("/two")
def cam2():
    return render_template("cam2.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running
    camera_running = False
    return ('', 204)

if __name__ == "__main__":
    app.run()
