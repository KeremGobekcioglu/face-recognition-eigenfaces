import cv2
import numpy as np
from scipy.ndimage import convolve
import os

def bgr_to_hsv(image):
    # Create an empty array for the HSV image
    hsv_image = np.zeros_like(image, dtype=np.float32)
    
    # Iterate over each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j] / 255.0  # Normalize the BGR values to [0, 1]
            
            max_val = max(b, g, r)
            min_val = min(b, g, r)
            delta = max_val - min_val
            
            # Calculate Hue
            if delta == 0:
                h = 0
            elif max_val == r:
                h = (60 * ((g - b) / delta) + 360) % 360
            elif max_val == g:
                h = (60 * ((b - r) / delta) + 120) % 360
            elif max_val == b:
                h = (60 * ((r - g) / delta) + 240) % 360
            
            # Calculate Saturation
            s = 0 if max_val == 0 else (delta / max_val)
            
            # Calculate Value
            v = max_val
            
            # Store the HSV values
            hsv_image[i, j] = [h, s, v]
    
    # Convert HSV values to the range [0, 255]
    hsv_image[:, :, 0] = hsv_image[:, :, 0] / 2  # Hue range [0, 180]
    hsv_image[:, :, 1:] *= 255  # Saturation and Value range [0, 255]
    
    return hsv_image.astype(np.uint8)

def bgr_to_ycrcb(image):
    # Create an empty array for the YCrCb image
    ycrcb_image = np.zeros_like(image, dtype=np.float32)
    
    # Iterate over each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j]
            
            # Calculate Y, Cr, and Cb
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cr = (r - y) * 0.713 + 128
            cb = (b - y) * 0.564 + 128
            
            # Store the YCrCb values
            ycrcb_image[i, j] = [y, cr, cb]
    
    # Convert YCrCb values to the range [0, 255]
    ycrcb_image = np.clip(ycrcb_image, 0, 255)
    
    return ycrcb_image.astype(np.uint8)

def in_range(image, lower_bound, upper_bound):
    # Create an empty mask with the same height and width as the image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Iterate over each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Check if the pixel values are within the specified range
            if all(lower_bound <= image[i, j]) and all(image[i, j] <= upper_bound):
                mask[i, j] = 255  # Set the mask pixel to white
            else:
                mask[i, j] = 0  # Set the mask pixel to black
    
    return mask

def get_structuring_element(shape, ksize):
    if shape != 'ellipse':
        raise ValueError("Only 'ellipse' shape is supported in this implementation")
    
    rows, cols = ksize
    kernel = np.zeros((rows, cols), dtype=np.uint8)
    
    center_x, center_y = cols // 2, rows // 2
    axes_x, axes_y = cols / 2, rows / 2
    
    for i in range(rows):
        for j in range(cols):
            if ((j - center_x) ** 2) / (axes_x ** 2) + ((i - center_y) ** 2) / (axes_y ** 2) <= 1:
                kernel[i, j] = 1
    
    return kernel

def bitwise_and(mask1, mask2):
    # Ensure both masks have the same shape
    assert mask1.shape == mask2.shape, "Masks must have the same shape"
    
    # Create an empty mask with the same shape
    result_mask = np.zeros_like(mask1, dtype=np.uint8)
    
    # Iterate over each pixel
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            # Perform bitwise AND operation
            result_mask[i, j] = mask1[i, j] & mask2[i, j]
    
    return result_mask

def erode(image, kernel, iterations=1):
    # Get the dimensions of the image and kernel
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Pad the image to handle borders
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
    
    for _ in range(iterations):
        # Create a copy of the image to store the result
        eroded_image = np.copy(image)
        
        # Iterate over each pixel in the image
        for i in range(img_h):
            for j in range(img_w):
                # Extract the region of interest
                roi = padded_image[i:i + k_h, j:j + k_w]
                
                # Apply the kernel (structuring element)
                if np.all(roi[kernel == 1] == 255):
                    eroded_image[i, j] = 255
                else:
                    eroded_image[i, j] = 0
        
        # Update the padded image for the next iteration
        padded_image = np.pad(eroded_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
    
    return eroded_image

def dilate(image, kernel, iterations=1):
    # Get the dimensions of the image and kernel
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Pad the image to handle borders
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    for _ in range(iterations):
        # Create a copy of the image to store the result
        dilated_image = np.copy(image)
        
        # Iterate over each pixel in the image
        for i in range(img_h):
            for j in range(img_w):
                # Extract the region of interest
                roi = padded_image[i:i + k_h, j:j + k_w]
                
                # Apply the kernel (structuring element)
                if np.any(roi[kernel == 1] == 255):
                    dilated_image[i, j] = 255
                else:
                    dilated_image[i, j] = 0
        
        # Update the padded image for the next iteration
        padded_image = np.pad(dilated_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    return dilated_image

def gaussian_kernel(size, sigma=1.0):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_filter(image, sigma=1.0):
    """Applies Gaussian filter to the image."""
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(size, sigma)
    return convolve(image, kernel)

def sobel(image, axis):
    """Applies Sobel filter to the image to compute the gradient."""
    if axis == 0:
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif axis == 1:
        kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    else:
        raise ValueError("Axis must be 0 (x) or 1 (y)")
    return convolve(image, kernel)

def canny(image, low_threshold, high_threshold):
    # Step 1: Apply Gaussian blur
    blurred_image = gaussian_filter(image, sigma=1.4)
    
    # Step 2: Compute gradient intensity and direction
    grad_x = sobel(blurred_image, axis=0)
    grad_y = sobel(blurred_image, axis=1)
    gradient_magnitude = np.hypot(grad_x, grad_y)
    gradient_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    gradient_direction[gradient_direction < 0] += 180
    
    # Step 3: Apply non-maximum suppression
    nms_image = np.zeros_like(gradient_magnitude)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            angle = gradient_direction[i, j]
            q = 255
            r = 255
            
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]
            
            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                nms_image[i, j] = gradient_magnitude[i, j]
            else:
                nms_image[i, j] = 0
    
    # Step 4: Apply double threshold
    strong_edges = (nms_image > high_threshold).astype(np.uint8)
    weak_edges = ((nms_image >= low_threshold) & (nms_image <= high_threshold)).astype(np.uint8)
    
    # Step 5: Track edges by hysteresis
    edges = np.zeros_like(image, dtype=np.uint8)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if strong_edges[i, j]:
                edges[i, j] = 255
            elif weak_edges[i, j]:
                if (strong_edges[i + 1, j - 1:j + 2].any() or
                    strong_edges[i - 1, j - 1:j + 2].any() or
                    strong_edges[i, [j - 1, j + 1]].any()):
                    edges[i, j] = 255
    
    return edges

def find_contours(image):
    contours = []
    visited = np.zeros_like(image, dtype=bool)
    
    def is_valid(x, y):
        return 0 <= x < image.shape[0] and 0 <= y < image.shape[1]
    
    def trace_contour(start):
        contour = []
        stack = [start]
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        
        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            contour.append((x, y))
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and image[nx, ny] == 255 and not visited[nx, ny]:
                    stack.append((nx, ny))
        
        return contour
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 255 and not visited[i, j]:
                contour = trace_contour((i, j))
                if contour:
                    contours.append(np.array(contour))
    
    return contours, None

def bounding_rect(contour):
    """
    Calculate the bounding rectangle for a given contour.

    Args:
        contour (list of tuples): A list of (x, y) points representing the contour.

    Returns:
        tuple: (x_min, y_min, width, height) of the bounding rectangle.
    """
    contour = np.array(contour)  # Convert the contour to a NumPy array
    x_min = np.min(contour[:, 1])
    y_min = np.min(contour[:, 0])
    x_max = np.max(contour[:, 1])
    y_max = np.max(contour[:, 0])
    return x_min, y_min, x_max - x_min, y_max - y_min


def bgr_to_gray(image):
    # Create an empty array for the grayscale image
    gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Iterate over each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_image[i, j] = gray
    
    return gray_image

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


def crop_face_using_edges(image_path, output_path):
    """
    Detect and crop the face using edges from the skin mask.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the cropped face.
    """
    # Step 1: Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Convert the image to HSV and YCbCr color spaces
    hsv_image = bgr_to_hsv(image)
    ycbcr_image = bgr_to_ycrcb(image)

    # Step 2: Define skin color thresholds
    # HSV thresholds for skin color
    lower_hsv = np.array([0, 30, 50], dtype=np.uint8)
    upper_hsv = np.array([50, 255, 255], dtype=np.uint8)
    mask_hsv = in_range(hsv_image, lower_hsv, upper_hsv)

    # YCbCr thresholds for skin color
    lower_ycbcr = np.array([0, 128, 80], dtype=np.uint8)
    upper_ycbcr = np.array([255, 180, 135], dtype=np.uint8)
    mask_ycbcr = in_range(ycbcr_image, lower_ycbcr, upper_ycbcr)

    # Combine the masks
    skin_mask = bitwise_and(mask_hsv, mask_ycbcr)

    # Step 3: Morphological operations to remove noise
    kernel = get_structuring_element('ellipse', (5, 5))
    skin_mask = erode(skin_mask, kernel, iterations=2)
    skin_mask = dilate(skin_mask, kernel, iterations=2)

    # Step 4: Detect edges in the skin mask
    edges = canny(skin_mask, 100, 150)

    # Step 5: Find contours of the edges
    contours, _ = find_contours(skin_mask)

    max_area = 0
    bounding_box = None

    for contour in contours:
        x, y, w, h = bounding_rect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            bounding_box = (x, y, w, h)

    if bounding_box:
        x, y, w, h = bounding_box
        # Narrow the cropping region horizontally
        narrow_factor_x = 0.1  # Adjust this value to control horizontal narrowing
        reduction_x = int(w * narrow_factor_x)
        x += reduction_x
        w -= 2 * reduction_x
        # Narrow the cropping region vertically
        narrow_factor_y = 0.1  # Adjust this value to control vertical narrowing
        reduction_y = int(h * narrow_factor_y)
        y += reduction_y
        h -= 2 * reduction_y
        # Ensure the new region is within bounds
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        cropped_region = image[y:y+h, x:x+w]

        gray = bgr_to_gray(cropped_region)

        # Step 4: Resize to fixed dimensions
        # resized = cv2.resize(gray, (100, 100))

        # # Step 5: Normalize pixel values to range 0-1
        # normalized = resized / 255.0

        # # Step 6: Flatten the image into a 1D vector
        # flattened = normalized.flatten()

        # Save the cropped region for further inspection
        cv2.imwrite(output_path, gray)
    else:
        print(f"No face detected in {image_path}.")





def process_images_in_folder(input_folder, output_folder):
    """
    Process all images in a folder, detect and crop faces, and save to another folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Skip non-image files
        if not (filename.lower().endswith(".png") or filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg")):
            continue

        print(f"Processing: {input_path}")
        try:
            output_path = os.path.join(output_folder, f"processed_{filename}")
            crop_face_using_edges(input_path, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processing completed. Processed images saved in: {output_folder}")

# Example usage
input_folder = "./faces_dataset_black"  # Folder containing input images
output_folder = "./output_images2"  # Folder to save processed images

process_images_in_folder(input_folder, output_folder)


