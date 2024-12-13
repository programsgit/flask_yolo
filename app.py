from flask import Flask, render_template, request, redirect, url_for
import torch
import cv2
import os

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(request.url)
    
    video_file = request.files['video']
    if video_file.filename == '' or not allowed_file(video_file.filename):
        return "File type not allowed", 400

    input_video_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    output_video_path = os.path.join(PROCESSED_FOLDER, 'output_video.mp4')
    
    # Save the uploaded video
    video_file.save(input_video_path)

    try:
        # Open the input video
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            results = model(frame)

            # Render results on the frame
            frame = results.render()[0]

            # Write the frame to the output video
            out.write(frame)

        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Optionally, delete the input video after processing
        os.remove(input_video_path)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

    return render_template('index.html', original_video=input_video_path, processed_video=output_video_path)

if __name__ == '__main__':
    app.run(debug=True)

