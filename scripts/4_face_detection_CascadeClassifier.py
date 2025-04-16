import requests
import cv2
import numpy as np
import time  # To add timestamps to filenames for uniqueness

url = 'http://192.168.0.101'  # URL of the video stream (e.g., MJPEG stream)
stream = requests.get(url, stream=True)# Open HTTP stream using requests
if not stream.ok:   # Check if the stream was successfully opened
    print(f"Failed to open stream at {url}")
    exit()
bytes_data = b''   # Initialize variables
ret, first_frame = False, None  # Read the first frame to initialize the background

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Load Haar cascade for face detection (you can use other classifiers as well)

# Loop through the stream data and process each chunk
for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk

    # Look for the start and end markers of JPEG images
    a = bytes_data.find(b'\xff\xd8')  # JPEG start marker
    b = bytes_data.find(b'\xff\xd9')  # JPEG end marker

    if a != -1 and b != -1:
        jpg = bytes_data[a:b + 2]  # Extract the JPEG frame
        bytes_data = bytes_data[b + 2:]  # Keep the remaining bytes

        # Convert the JPEG bytes to an image
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Convert the current frame to grayscale and blur it to reduce noise
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # Detect faces in the ROI
        # Draw rectangles around faces
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            print("Face detected!")

            # Save the image when a face is detected
            # timestamp = time.strftime("%Y%m%d_%H%M%S")  # Create a timestamp for the filename
            # filename = f"face_detected_{timestamp}.jpg"
            # cv2.imwrite(filename, frame)  # Save the frame as an image
            # print(f"Image saved as {filename}")

        cv2.imshow("Motion Detection Stream - Frame Differencing", frame) # Display the frame with motion detection and object detection
        first_frame = gray_frame # Update the first frame to the current frame for the next iteration
        # Exit the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
