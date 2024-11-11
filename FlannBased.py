import cv2
import numpy as np
import os
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture the static background
print("Capturing background... Please make sure the scene is empty.")
cv2.waitKey(2000)
ret, static_background = cap.read()
static_background_gray = cv2.cvtColor(static_background, cv2.COLOR_BGR2GRAY)
static_background_gray = cv2.GaussianBlur(static_background_gray, (21, 21), 0)

print("Background captured! Press SPACE to start capturing and matching ROI.")

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_sift_features_bf(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Preload all SIFT descriptors from the image directory into memory
def preload_sift_descriptors(dir_path):
    sift_data = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                _, descriptors = extract_sift_features(img)
                if descriptors is not None:
                    sift_data.append((image_path, descriptors))  # Store file path and descriptors
    return sift_data

# Load all descriptors in memory for faster access
sift_data_in_memory = preload_sift_descriptors("images")

matching_mode = False  # Flag for matching mode
last_match_time = 0  # Last time a match was attempted
best_directory_text = ""  # Text to display the best match directory

while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale and apply Gaussian blur
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Calculate absolute difference for background subtraction
    diff_frame = cv2.absdiff(static_background_gray, gray_frame)
    _, thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)

    # Capture the bottom-left region from the thresholded frame
    h, w = thresh_frame.shape[:2]
    roi = thresh_frame[h // 2:, :w // 2]  # bottom-left region

    # Display the segmented ROI
    cv2.imshow("Segmented ROI", roi)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Start matching mode when SPACE is pressed
    if key == ord(' '):
        matching_mode = True
        print("Matching mode enabled. Starting SIFT matching with BF...")

    # Perform matching every 1 second if in matching mode
    if matching_mode and (time.time() - last_match_time > 1):
        print("Matching captured ROI...")
        _, query_descriptors = extract_sift_features(roi)

        if query_descriptors is not None:
            # Find the best match from the preloaded descriptors
            best_match_count = 0
            best_match_path = "No matches found"
            
            for image_path, descriptors in sift_data_in_memory:
                matches = match_sift_features_bf(query_descriptors, descriptors)
                match_count = len(matches)
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_match_path = image_path
            
            # Extract subdirectory name from the best match path
            subdirectory_name = os.path.basename(os.path.dirname(best_match_path))
            best_directory_text = f"Best Match: {subdirectory_name} ({best_match_count} matches)"
            print(f"Best match found: {subdirectory_name} with {best_match_count} matches")
        else:
            best_directory_text = "Failed to extract features from the ROI."
            print("Failed to extract features from the ROI.")
        
        # Update the last match time
        last_match_time = time.time()
    
    # Display the best match text on the frame
    cv2.putText(frame, best_directory_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the updated frame with text
    cv2.imshow("Original Frame", frame)

    # Exit the loop when 'q' key is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
