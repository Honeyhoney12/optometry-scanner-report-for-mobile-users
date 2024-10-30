from flask import Flask, render_template, request, redirect, url_for, Response, session, jsonify
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Global variable to hold the latest redness percentage
latest_redness_percentage = 0.0

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_redness_percentage(eye_image):
    eye_image = eye_image.astype(float)
    b_channel, g_channel, r_channel = cv2.split(eye_image)
    total_intensity = r_channel + g_channel + b_channel
    redness = r_channel / (total_intensity + 1e-5)  # Avoid division by zero
    redness_percentage = np.mean(redness) * 100
    return redness_percentage

def generate_frames():
    global latest_redness_percentage
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    eye_image = roi_color[ey:ey + eh, ex:ex + ew]
                    latest_redness_percentage = get_redness_percentage(eye_image)  # Update global variable
                    cv2.putText(frame, f'Eye power: {latest_redness_percentage:.1f}%', (x, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['name'] = request.form['name']
        session['gender'] = request.form['gender']
        session['age'] = request.form['age']
        return redirect(url_for('scanner'))
    return render_template('index.html')

@app.route('/scanner')
def scanner():
    return render_template('scanner.html', name=session.get('name'), gender=session.get('gender'), age=session.get('age'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_redness')
def get_redness():
    global latest_redness_percentage
    return jsonify({"redness_percentage": latest_redness_percentage})
@app.route('/language', methods=['GET', 'POST'])
def language():
    if request.method == 'POST':
        session['language'] = request.form['language']  # Save selected language in session
        return redirect(url_for('index'))  # Redirect to the main page
    return render_template('lang.html')


if __name__ == '__main__':
    app.run(debug=True)
