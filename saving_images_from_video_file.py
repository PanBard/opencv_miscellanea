import cv2
import numpy as np
import time  # To add timestamps to filenames for uniqueness

# Path to your MP4 video file
video_path = 'movie.mp4'

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the FPS (frames per second) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize variables to capture and process the video
counter = 0  # for image numbering

print("To save an image, press 's'. Press 'q' to quit.")

while True:
    # Read the next frame from the video
    ret, frame = cap.read()

    if not ret:  # If the frame was not read successfully, exit the loop
        print("End of video or failed to read frame.")
        break
    
    # Show the frame
    cv2.imshow("Video Stream", frame)

    # Wait for the appropriate time based on the video's FPS
    key = cv2.waitKey(int(1000 / fps)) & 0xFF  # Delay in milliseconds to match FPS
    
    if key == ord('q'):  # Exit on pressing 'q'
        break
    elif key == ord('s'):  # Save the image on pressing 's'
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Create a timestamp for the filename
        filename = f"images/image_{counter}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)  # Save the frame as an image
        print(f"Image saved as {filename}")
        counter = counter + 1

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
