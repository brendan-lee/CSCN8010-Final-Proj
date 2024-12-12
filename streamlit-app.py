import os

import cv2
import streamlit as st
from ultralytics import YOLO


# Load two models
@st.cache_resource
def load_models():
  vehicle_model = YOLO("./models/cars.pt")
  traffic_light_model = YOLO("./models/traffic_light.pt")
  return vehicle_model, traffic_light_model


# Load the models
vehicle_model, traffic_light_model = load_models()

# Define labels for each model
vehicle_labels = ["Car"]
traffic_light_labels = ["Red Light", "Green Light"]


# Function to process video frame-by-frame
def process_video(video_path, output_path, target_fps=20):
  # Open the input video
  cap = cv2.VideoCapture(video_path)

  # Get video properties
  fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
  # Total number of frames
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
  fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Video codec

  # Create output video writer
  out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

  # Calculate the frame skipping interval
  frame_interval = max(1, int(fps / target_fps))
  # Initialize frame count
  frame_count = 0

  # Initialize progress bar
  progress_bar = st.progress(0)
  processed_frames = 0

  # Process video frame by frame
  while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if no more frames
    if not ret:
      break

    # process only every frame_interval frames
    if frame_count % frame_interval == 0:
      # Run vehicle detection using YOLO
      vehicle_results = vehicle_model(frame, conf=0.5)
      # Run traffic light detection using YOLO
      traffic_light_results = traffic_light_model(frame, conf=0.5)

      # Draw bounding boxes for detected vehicles
      for result in vehicle_results[0].boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        # Class label
        label = vehicle_labels[int(result.cls)]
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)  # Add label text

      # Draw bounding boxes for detected traffic lights

      # Initialize traffic light state
      traffic_light_state = None
      for result in traffic_light_results[0].boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        # Class label
        label = traffic_light_labels[int(result.cls)]
        if label == "Red Light":
          # Update state to red
          traffic_light_state = "Red Light"
        elif label == "Green Light":
          # Update state to green
          traffic_light_state = "Green Light"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Add label text
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

        # Add descriptive text at the bottom of the frame
      description = f"{traffic_light_state}, Vehicles Detected: {len(vehicle_results[0].boxes)}"
      cv2.putText(frame, description, (10, height - 20),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.8, (255, 255, 255), 2)

      # Write the modified frame to the output video
      out.write(frame)

    # Write only frames that match the interval
    frame_count += 1

    # Update progress bar
    processed_frames += 1
    # Ensure progress does not exceed 1.0
    progress = min(processed_frames / total_frames, 1.0)
    progress_bar.progress(int(progress * 100))

  # Release resources
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  progress_bar.progress(100)


# Streamlit application
st.title("Vehicle and Traffic Light Detection")
st.write(
    "Upload a video to detect vehicle positions and traffic light states, and view the processed video with annotations.")

# Upload video file
uploaded_video = st.file_uploader("Upload a video file",
                                  type=["mp4", "avi", "mov"])

if uploaded_video is not None:
  os.makedirs("tmp", exist_ok=True)
  input_path = "./tmp/input_video.mp4"
  output_path = "./tmp/output_video.mp4"

  # Save the uploaded video file locally
  with open(input_path, "wb") as f:
    f.write(uploaded_video.read())

  # Process the video and generate annotated output
  st.write("Processing video, please wait...")
  process_video(input_path, output_path, target_fps=16)

  st.video(output_path)
