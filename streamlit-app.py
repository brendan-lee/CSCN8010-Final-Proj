import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

# Load two YOLO models: one for vehicle detection and another for traffic light detection
@st.cache_resource
def load_models():
    vehicle_model = YOLO("path_to_vehicle_model.pt")  # Replace with the path to your vehicle model
    traffic_light_model = YOLO("path_to_traffic_light_model.pt")  # Replace with the path to your traffic light model
    return vehicle_model, traffic_light_model

# Load the models
vehicle_model, traffic_light_model = load_models()

# Function to process video frame-by-frame
def process_video(video_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Create output video writer

    # Define labels for each model
    vehicle_labels = ["Car", "Truck", "Bus"]  # Replace with your vehicle model's labels
    traffic_light_labels = ["Red Light", "Green Light"]  # Replace with your traffic light model's labels

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Break the loop if no more frames

        # Run vehicle detection using YOLO
        vehicle_results = vehicle_model(frame, conf=0.5)  # Confidence threshold 0.5
        # Run traffic light detection using YOLO
        traffic_light_results = traffic_light_model(frame, conf=0.5)

        # Draw bounding boxes for detected vehicles
        for result in vehicle_results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            label = vehicle_labels[int(result.cls)]  # Class label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add label text

        # Draw bounding boxes for detected traffic lights
        traffic_light_state = None  # Initialize traffic light state
        for result in traffic_light_results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            label = traffic_light_labels[int(result.cls)]  # Class label
            if label == "Red Light":
                traffic_light_state = "Red Light"  # Update state to red
            elif label == "Green Light":
                traffic_light_state = "Green Light"  # Update state to green
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw rectangle
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Add label text

        # Add descriptive text at the bottom of the frame
        description = f"{traffic_light_state}, Vehicles Detected: {len(vehicle_results[0].boxes)}"
        cv2.putText(frame, description, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Write the modified frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

# Streamlit application
st.title("Vehicle and Traffic Light Detection")
st.write("Upload a video to detect vehicle positions and traffic light states, and view the processed video with annotations.")

# Upload video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    input_path = "input_video.mp4"  # Temporary path for the input video
    output_path = "output_video.mp4"  # Path for the processed video

    # Save the uploaded video file locally
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    # Process the video and generate annotated output
    st.write("Processing video, please wait...")
    process_video(input_path, output_path)

    # Display the processed video
    st.video(output_path)
