import cv2
import numpy as np
import os
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Ensure the save directory exists
save_dir = "images/ok"
os.makedirs(save_dir, exist_ok=True)

# Capture the static background
print("Capturing background... Please make sure the scene is empty.")
cv2.waitKey(2000)
ret, static_background = cap.read()

# Convert the static background to grayscale
static_background_gray = cv2.cvtColor(static_background, cv2.COLOR_BGR2GRAY)
static_background_gray = cv2.GaussianBlur(static_background_gray, (21, 21), 0)

print("Background captured! Starting background subtraction and displaying segmented ROI.")

capture_mode = False  # Flag to indicate if we are in capture mode
capture_start_time = None  # Time when capturing started

while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale and apply Gaussian blur
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Calculate the absolute difference between the static background and the current frame
    diff_frame = cv2.absdiff(static_background_gray, gray_frame)
    _, thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)

    # Capture the bottom-left region from the thresholded frame (adjusted for left-bottom)
    h, w = thresh_frame.shape[:2]
    region = thresh_frame[h // 2:, :w // 2]  # Adjusted to capture the left half (bottom-left)

    # Display the original frame and the segmented ROI
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Segmented ROI", region)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Start capture mode when "1" key is pressed
    if key == ord('1'):
        capture_mode = True
        capture_start_time = time.time()

    # Capture images every 0.5 seconds for 10 seconds
    if capture_mode:
        elapsed_time = time.time() - capture_start_time
        if elapsed_time <= 15:  # Capture for 10 seconds
            if int(elapsed_time * 2) % 0.25 == 0:  # Capture every 0.5 seconds
                # Save the segmented image
                timestamp = int(time.time() * 1000)  # Use timestamp to create unique file name
                filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, region)
                time.sleep(0.5)  # Wait for 0.5 seconds before the next capture
        else:
            capture_mode = False  # Stop capturing after 10 seconds

    # Exit the loop when 'q' key is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
