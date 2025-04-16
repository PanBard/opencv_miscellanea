import requests
import cv2
import numpy as np
import time  # To add timestamps to filenames for uniqueness

#Basic Approach: Frame Differencing
#This method compares two consecutive frames and calculates the difference. If there is a significant change in pixel values, it indicates movement.

url = 'http://192.168.0.101'  # URL of the video stream (e.g., MJPEG stream)
stream = requests.get(url, stream=True)# Open HTTP stream using requests
if not stream.ok:   # Check if the stream was successfully opened
    print(f"Failed to open stream at {url}")
    exit()
bytes_data = b''   # Initialize variables
ret, first_frame = False, None  # Read the first frame to initialize the background

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

        if not ret:
            # Initialize the first frame if it's not already done
            first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)
            ret = True
            continue  # Skip motion detection for the first frame

        # Convert the current frame to grayscale and blur it to reduce noise
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # Compute the absolute difference between the current and first frame
        frame_diff = cv2.absdiff(first_frame, gray_frame)

        # Threshold the difference to get the motion areas
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Find contours (motion areas)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False  # Flag to track motion detection status

        # Draw rectangles around the moving areas and print message when motion is detected
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Ignore small contours (noise)
                continue

            # Get the bounding box of the moving object
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # If motion is detected, print "motion detected"
            if not motion_detected:
                # print("Motion detected!")
                motion_detected = True  # Set the flag to True once motion is detected

                # # Save the image when motion is detected
                # timestamp = time.strftime("%Y%m%d_%H%M%S")  # Create a timestamp for the filename
                # filename = f"motion_detected_{timestamp}.jpg"
                # cv2.imwrite(filename, frame)  # Save the frame as an image
                # print(f"Image saved as {filename}")


        # Display the frame with motion detection and object detection
        cv2.imshow("Motion Detection Stream - Frame Differencing", frame)

        # Update the first frame to the current frame for the next iteration
        first_frame = gray_frame

        # Exit the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
