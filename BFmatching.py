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

def match_sift_features(query_descriptors, stored_descriptors):
    index_params = dict(algorithm=1, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(query_descriptors, stored_descriptors, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return len(good_matches)

# Precompute and store descriptors for each image in the directory
def load_descriptors(dir_path):
    descriptors_dict = {}
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                _, descriptors = extract_sift_features(img)
                if descriptors is not None:
                    descriptors_dict[image_path] = descriptors
    return descriptors_dict

print("Loading descriptors for matching...")
image_descriptors = load_descriptors("images")
print(f"Loaded descriptors for {len(image_descriptors)} images.")

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

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Start matching mode when SPACE is pressed
    if key == ord(' '):
        matching_mode = True
        print("Matching mode enabled. Starting SIFT matching...")

    # Perform matching every 1 second if in matching mode
    if matching_mode and (time.time() - last_match_time > 1):
        print("Matching captured ROI...")
        _, query_descriptors = extract_sift_features(roi)

        if query_descriptors is not None:
            match_results = []
            for image_path, descriptors in image_descriptors.items():
                match_count = match_sift_features(query_descriptors, descriptors)
                match_results.append((match_count, image_path))

            # Sort by number of good matches and select top 10
            match_results.sort(key=lambda x: x[0], reverse=True)
            best_matches = match_results[:10]

            # Identify the directory of the best match
            if best_matches:
                best_directory = os.path.dirname(best_matches[0][1])
                best_directory_text = f"Best Match Directory: {best_directory}"
                print(f"Top 10 matches found in directory: {best_directory}")
                for i, (match_count, img_path) in enumerate(best_matches):
                    print(f"Rank {i+1}: {img_path} with {match_count} good matches")
            else:
                best_directory_text = "No good matches found."
                print("No good matches found.")
        else:
            best_directory_text = "Failed to extract features from the ROI."
            print("Failed to extract features from the ROI.")
        
        # Update the last match time
        last_match_time = time.time()
    
    # Display the best match directory text on the frame
    cv2.putText(frame, best_directory_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the updated frame with text
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Segmented ROI", roi)

    # Exit the loop when 'q' key is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
