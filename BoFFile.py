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

# Number of clusters (visual words)
num_clusters = 15

# Collect all descriptors for clustering
all_descriptors = []
for _, descriptors in sift_data_in_memory:
    if descriptors is not None:
        all_descriptors.append(descriptors)

# Stack all descriptors into a single numpy array for k-means clustering
all_descriptors = np.vstack(all_descriptors).astype(np.float32)

# Apply k-means clustering using OpenCV
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, visual_words = cv2.kmeans(
    all_descriptors, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

# Compute a BoF histogram for each image using the visual words from k-means
def compute_bof_histogram(descriptors, visual_words):
    histogram = np.zeros(num_clusters)
    for descriptor in descriptors:
        diff = np.linalg.norm(visual_words - descriptor, axis=1)
        closest_word = np.argmin(diff)
        histogram[closest_word] += 1
    return histogram
def apply_gamma_transformation(image, gamma=1.5):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    
    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, table)
# Create BoF histograms for each image and store them in a dictionary
bof_histograms = {}
for image_path, descriptors in sift_data_in_memory:
    if descriptors is not None:
        bof_histograms[image_path] = compute_bof_histogram(descriptors, visual_words)

matching_mode = False
last_match_time = 0
best_match_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21,21), 0)
    # gray_frame=apply_gamma_transformation(gray_frame)
    diff_frame = cv2.absdiff(static_background_gray, gray_frame)
    _, thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)

    h, w = thresh_frame.shape[:2]
    roi = thresh_frame[h // 2:, :w // 2]  # bottom-left region

    cv2.imshow("Segmented ROI", roi)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        matching_mode = True
        print("Matching mode enabled. Starting BoF matching...")

    if matching_mode and (time.time() - last_match_time > 1):
        print("Matching captured ROI...")
        _, query_descriptors = extract_sift_features(roi)

        if query_descriptors is not None:
            query_histogram = compute_bof_histogram(query_descriptors, visual_words)

            best_match_score = float('inf')
            best_match_path = "No matches found"
            
            for image_path, histogram in bof_histograms.items():
                score = np.sum(((query_histogram - histogram) ** 2) / (query_histogram + histogram + 1e-10))
                if score < best_match_score:
                    best_match_score = score
                    best_match_path = os.path.basename(os.path.dirname(image_path))  # Get only subdirectory name
            
            best_match_text = f"Best Match: {best_match_path} (Score: {best_match_score:.2f})"
            print(f"Best match found: {best_match_path} with score: {best_match_score:.2f}")
        else:
            best_match_text = "Failed to extract features from the ROI."
            print("Failed to extract features from the ROI.")
        
        last_match_time = time.time()
    
    cv2.putText(frame, best_match_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Original Frame", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
