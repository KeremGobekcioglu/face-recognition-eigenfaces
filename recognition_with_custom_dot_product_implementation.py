import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def resize(src, dsize, interpolation=cv2.INTER_LINEAR):
    src_height, src_width = src.shape[:2]
    dst_width, dst_height = dsize
    dst = np.zeros((dst_height, dst_width), dtype=src.dtype)
    
    for i in range(dst_height):
        for j in range(dst_width):
            src_x = j * (src_width / dst_width)
            src_y = i * (src_height / dst_height)
            src_x0 = int(np.floor(src_x))
            src_y0 = int(np.floor(src_y))
            src_x1 = min(src_x0 + 1, src_width - 1)
            src_y1 = min(src_y0 + 1, src_height - 1)
            
            dx = src_x - src_x0
            dy = src_y - src_y0
            
            dst[i, j] = (1 - dx) * (1 - dy) * src[src_y0, src_x0] + \
                        dx * (1 - dy) * src[src_y0, src_x1] + \
                        (1 - dx) * dy * src[src_y1, src_x0] + \
                        dx * dy * src[src_y1, src_x1]
    
    return dst
def custom_mean(array, axis=None):
    if axis is None:
        return sum(array) / len(array)
    else:
        return np.sum(array, axis=axis) / array.shape[axis]

def create_subdirectories(base_dir="results"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    subdirectories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    run_number = len(subdirectories)
    subdirectory = os.path.join(base_dir, f"results_{run_number}")
    os.makedirs(subdirectory)
    return subdirectory

def save_image_to_file(image, filename):
    cv2.imwrite(filename, image)

def custom_dot(a, b):
    result = []
    for a_row in tqdm(a, desc="Computing dot product (outer loop)"):
        row_result = []
        for b_col in tqdm(zip(*b), desc="Computing dot product (inner loop)", leave=False):
            row_result.append(sum(x * y for x, y in zip(a_row, b_col)))
        result.append(row_result)
    return np.array(result)

def normalize(src, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
    if dst is None:
        dst = np.zeros_like(src)
    
    if norm_type == cv2.NORM_MINMAX:
        min_val = np.min(src)
        max_val = np.max(src)
        dst = (src - min_val) * (beta - alpha) / (max_val - min_val) + alpha
    else:
        raise NotImplementedError("Only NORM_MINMAX is implemented")
    
    return dst

# Add labels to the image
def add_labels_to_image(image, labels, positions, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=1):
    for label, position in zip(labels, positions):
        cv2.putText(image, label, position, font, font_scale, color, thickness, cv2.LINE_AA)

# This function loads our dataset of Olivetti faces and resizes them to a target size.
# It returns a list of image paths and their corresponding labels.
def load_faces(root_dir, target_size=(100,100)):
    image_paths = []
    labels = []

    print(f"Loading images from: {root_dir}")
    # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: The directory {root_dir} does not exist.")
        return image_paths, labels

    for person_dir in tqdm(os.listdir(root_dir), desc="Processing directories"):
        person_path = os.path.join(root_dir, person_dir)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(person_path, filename)
                    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Error loading image: {full_path}")
                        continue
                    resized_image = resize(image, target_size)
                    save_path = os.path.join(person_path, filename)
                    cv2.imwrite(save_path, resized_image)
                    image_paths.append(save_path)
                    labels.append(person_dir)
                    print(f"Loaded and resized: {person_dir} - {filename}")
    print(f"Total images loaded: {len(image_paths)}")
    return image_paths, labels

# Step 2: Preprocess Images (Load, Flatten, Normalize)
def preprocess_images(image_paths):
    dataset = []
    for path in tqdm(image_paths, desc="Preprocessing images"):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading image: {path}")
            continue
        # Flatten and normalize the image
        flattened = (image.flatten() / 255.0)
        dataset.append(flattened)
    return np.array(dataset, dtype=np.float64)

# Step 3: Compute Mean Face
def compute_mean_face(dataset):
    mean_face = custom_mean(dataset, axis=0)
    
    # Reshape the mean face to its original dimensions (50x50)
    mean_face_image = mean_face.reshape((100,100))
    
    # Normalize the mean face image to the range [0, 255]
    mean_face_image = normalize(mean_face_image, norm_type=cv2.NORM_MINMAX)
    
    # Convert to uint8 type
    mean_face_image = mean_face_image.astype(np.uint8)
    
    return mean_face, mean_face_image

# Step 4: Center Dataset
def center_dataset(dataset, mean_face):
    return dataset - mean_face

# Step 5: Compute Covariance Matrix
def compute_covariance_matrix(centered_data):
    num_images, num_features = centered_data.shape
    return custom_dot(centered_data.T, centered_data) / num_images

# Step 6: Perform Eigen Decomposition
def compute_eigenfaces(centered_data, covariance_matrix, num_eigenfaces):
    print("Computing eigenfaces")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, :num_eigenfaces]
    eigenfaces = custom_dot(eigenvectors.T, centered_data)
    return eigenfaces, eigenvalues[:num_eigenfaces]

def display_eigenfaces(eigenfaces, num_eigenfaces, image_shape=(100,100), results_subdirectory=None):
    # Calculate the grid size dynamically
    grid_cols = int(np.ceil(np.sqrt(num_eigenfaces)))
    grid_rows = int(np.ceil(np.sqrt(num_eigenfaces)))
    
    # Create a blank canvas to display the eigenfaces
    grid_height = grid_rows * image_shape[0]
    grid_width = grid_cols * image_shape[1]
    canvas = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    for i in range(num_eigenfaces):
        row = i // grid_cols
        col = i % grid_cols
        eigenface = eigenfaces[i].reshape(image_shape)
        eigenface = normalize(eigenface, norm_type=cv2.NORM_MINMAX)
        eigenface = eigenface.astype(np.uint8)
        canvas[row * image_shape[0]:(row + 1) * image_shape[0], col * image_shape[1]:(col + 1) * image_shape[1]] = eigenface
    
    # Display the canvas with all eigenfaces
    cv2.imshow("Eigenfaces", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the canvas with all eigenfaces
    if results_subdirectory:
        eigenfaces_path = os.path.join(results_subdirectory, "eigenfaces.jpg")
        cv2.imwrite(eigenfaces_path, canvas)

# Step 7: Project Faces Onto Eigenfaces
def project_faces(centered_data, eigenfaces):
    return custom_dot(centered_data, eigenfaces.T)

def save_results(results, filename):
    with open(filename, 'w') as f:
        for result in results:
            f.write(f"{result}\n")

# Step 8: Recognize Test Face
def recognize_face(test_face, mean_face, eigenfaces, projected_faces, labels, original_faces, fixed_threshold=300):
    centered_test_face = test_face - mean_face
    centered_test_face = centered_test_face.reshape(1, -1)
    projected_test_face = custom_dot(centered_test_face, eigenfaces.T)
    distances = np.linalg.norm(projected_faces - projected_test_face, axis=1)
    min_distance = np.min(distances)
    recognized_label = labels[np.argmin(distances)]
    closest_face_index = np.argmin(distances)
    threshold = np.percentile(distances, 10)
    # Use a fixed threshold
    # threshold = fixed_threshold
    
    # Debugging: Print distances and threshold
    print(f"Threshold: {threshold}")
    print(f"Min distance: {min_distance}")
    
    if min_distance > threshold:
        recognized_label = "unknown"
    else:
        recognized_label = labels[closest_face_index]
    # Reconstruct the projected face
    # reconstructed_face = np.dot(projected_test_face, eigenfaces) + mean_face
    reconstructed_face = custom_dot(projected_test_face, eigenfaces) + mean_face
    
    # Display the test face, mean face, closest face, and reconstructed face side by side
    test_face_image = test_face.reshape((100,100))
    mean_face_image = mean_face.reshape((100,100))
    closest_face = original_faces[closest_face_index].reshape((100,100))
    reconstructed_face_image = reconstructed_face.reshape((100,100))
    
    # Normalize the images to the range [0, 255]
    test_face_image = normalize(test_face_image, norm_type=cv2.NORM_MINMAX)
    mean_face_image = normalize(mean_face_image, norm_type=cv2.NORM_MINMAX)
    closest_face = normalize(closest_face, norm_type=cv2.NORM_MINMAX)
    reconstructed_face_image = normalize(reconstructed_face_image, norm_type=cv2.NORM_MINMAX)
    
    # Convert to uint8 type
    test_face_image = test_face_image.astype(np.uint8)
    mean_face_image = mean_face_image.astype(np.uint8)
    closest_face = closest_face.astype(np.uint8)
    reconstructed_face_image = reconstructed_face_image.astype(np.uint8)
    
    # Resize images to be larger for better visibility
    test_face_image = resize(test_face_image, (200, 200))
    mean_face_image = resize(mean_face_image, (200, 200))
    closest_face = resize(closest_face, (200, 200))
    reconstructed_face_image = resize(reconstructed_face_image, (200, 200))
    
    # Concatenate images horizontally
    combined_image = np.hstack((test_face_image, mean_face_image, closest_face, reconstructed_face_image))
    
    # Add labels to the combined image
    labels = ["Test Face", "Mean Face", "Closest Face", "Reconstructed Face"]
    positions = [(10, 20), (210, 20), (410, 20), (610, 20)]
    add_labels_to_image(combined_image, labels, positions)
    
    # Save the result
    return recognized_label , combined_image

# Paths
# olivetti_faces_dir = "olivetti_faces"  # Directory with Olivetti faces
olivetti_faces_dir = "cropped_faces"

# Step 1: Load and Resize Olivetti Faces
image_paths, labels = load_faces(olivetti_faces_dir)

# Step 2: Preprocess Dataset
dataset = preprocess_images(image_paths)

# Split the dataset into training and test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2)

# Preprocess training and test sets
train_dataset = preprocess_images(train_paths)
test_dataset = preprocess_images(test_paths)

# Step 3: Compute Mean Face
mean_face, mean_face_image = compute_mean_face(train_dataset)

# Step 4: Center the Dataset
centered_train_data = center_dataset(train_dataset, mean_face)

# Step 5: Compute Covariance Matrix
covariance_matrix = compute_covariance_matrix(centered_train_data)
print("Covariance matrix is calculated ")
# Step 6: Compute Eigenfaces
num_eigenfaces = min(10, train_dataset.shape[0])
eigenfaces, eigenvalues = compute_eigenfaces(centered_train_data, covariance_matrix, num_eigenfaces)

# Create a subdirectory for this run's results
results_subdirectory = create_subdirectories()

# Save mean face
mean_face_path = os.path.join(results_subdirectory, "mean_face.jpg")
cv2.imwrite(mean_face_path, mean_face_image)

# Display and save eigenfaces
display_eigenfaces(eigenfaces, num_eigenfaces, results_subdirectory=results_subdirectory)

# Step 7: Project Faces Onto Eigenfaces
projected_train_faces = project_faces(centered_train_data, eigenfaces)

# Test the recognition on the test set
correct_predictions = 0
unknown = 0
count = 1
results = []
for test_image_path, true_label in tqdm(zip(test_paths, test_labels), desc="Recognizing faces", total=len(test_paths)):
    test_face_dataset = preprocess_images([test_image_path])
    recognized_label , combined_image= recognize_face(
        test_face_dataset[0], 
        mean_face, 
        eigenfaces, 
        projected_train_faces, 
        train_labels,
        train_dataset,
        fixed_threshold=300  # Set a fixed threshold
    )
    if recognized_label == "unknown":
        print(f"Unknown face: {test_image_path}")
        result = f"Test image: {true_label}, True label: {true_label}, Recognized label: unknown , UNSUCCESFULL"
        output_path = os.path.join(results_subdirectory, f"{count}_{true_label}_recognized_as_unknown_result.jpg")
        unknown += 1
    elif recognized_label == true_label:
        print(f"TRUE")
        result = f"Test image: {true_label}, True label: {true_label}, Recognized label: {recognized_label} , SUCCESFULL"
        output_path = os.path.join(results_subdirectory, f"{count}_{true_label}_recognized_as_{true_label}_result.jpg")
        correct_predictions += 1
    else:
        print(f"True label: {true_label}, Recognized as: {recognized_label}")
        result = f"Test image: {true_label}, True label: {true_label}, Recognized label: {recognized_label} , UNSUCCESFULL"
        output_path = os.path.join(results_subdirectory, f"{count}_{true_label}_recognized_as_{recognized_label}_result.jpg")
    count += 1
    # Save the result
    cv2.imwrite(output_path, combined_image)
    results.append(result)

# Save all results to a file

# Calculate and print the recognition accuracy
accuracy = correct_predictions / len(test_paths)
print(f"Recognition accuracy: {accuracy * 100:.2f}%")
print(f"Unknown recognition: {unknown} out of {len(test_paths)}")
results.append(f"Recognition accuracy: {accuracy * 100:.2f}%")
results.append(f"Unknown recognition: {unknown} out of {len(test_paths)}")
results_file = os.path.join(results_subdirectory, "results.txt")
save_results(results, results_file)