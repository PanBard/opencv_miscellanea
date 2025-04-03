import requests
import cv2
import numpy as np
import time  # To add timestamps to filenames for uniqueness

# URL of the video stream (e.g., MJPEG stream)
url = 'http://192.168.0.101'

# Open HTTP stream using requests
stream = requests.get(url, stream=True)

# Initialize variables to capture and process the video
bytes_data = b''
counter = 0 # for image numbering


print("To save image press 's' ")

while True:
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # JPEG start marker
        b = bytes_data.find(b'\xff\xd9')  # JPEG end marker
        
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]  # Extract the JPEG frame
            bytes_data = bytes_data[b+2:]  # Keep the remaining bytes

            # Convert the JPEG bytes to an image
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # Show the frame
            cv2.imshow("Stream", frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Exit on pressing 'q'
                break
            elif key == ord('s'):  # Save the image on pressing 's'
                timestamp = time.strftime("%Y%m%d_%H%M%S")  # Create a timestamp for the filename
                filename = f"images/image_{counter}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)  # Save the frame as an image
                print(f"Image saved as {filename}")
                counter = counter + 1

    if key == ord('q'):  # Break the loop if 'q' is pressed
        break

cv2.destroyAllWindows()
