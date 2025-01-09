import cv2
import mediapipe as mp
from flask import Flask, render_template, Response

app = Flask(__name__)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Skeleton overlay function
def overlay_skeleton(frame, landmarks):
    # Load a realistic skeleton image (pre-scaled for simplicity)
    skeleton_img = cv2.imread("static/skeleton.png", cv2.IMREAD_UNCHANGED)
    skeleton_height, skeleton_width = skeleton_img.shape[:2]

    for i, lm in enumerate(landmarks.landmark):
        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        if i == 0:  # Example: Place skeleton at nose position
            overlay_x, overlay_y = x - skeleton_width // 2, y - skeleton_height // 2
            # Overlay skeleton image
            frame[overlay_y:overlay_y + skeleton_height, overlay_x:overlay_x + skeleton_width] = skeleton_img

    return frame

# Flask video stream generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Flip for a mirror effect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame = overlay_skeleton(frame, results.pose_landmarks)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
