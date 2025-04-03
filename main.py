import requests
import cv2
import numpy as np

# URL of the video stream (e.g., MJPEG stream)
url = 'http://192.168.0.101'

# Open HTTP stream using requests
stream = requests.get(url, stream=True)

# Initialize variables to capture and process the video
bytes_data = b''

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

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
