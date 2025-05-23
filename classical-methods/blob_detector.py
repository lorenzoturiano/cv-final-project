import argparse
import cv2
import numpy as np
import json
from astropy.io import fits
from auto_stretch import apply_stretch

import math

def calculate_median_circle(image, cx, cy, radius, channels):
        """
        Calculate the median value of a circle for the specified channels.

        Args:
            image (np.ndarray): The image array.
            cx (int): X-coordinate of the circle center.
            cy (int): Y-coordinate of the circle center.
            radius (int): Radius of the circle.
            channels (list): List of channel indices to process.

        Returns:
            float: The overall median value across specified channels.
        """
        values = []
        for c in channels:
            y_min = max(cy - radius, 0)
            y_max = min(cy + radius + 1, image.shape[0])
            x_min = max(cx - radius, 0)
            x_max = min(cx + radius + 1, image.shape[1])

            if image.ndim == 2:
                # Grayscale image (2D)
                roi = image[y_min:y_max, x_min:x_max]
            elif image.ndim == 3:
                if image.shape[2] ==1:
                    # Grayscale image with single channel
                    roi = image[y_min:y_max, x_min:x_max, 0]
                elif image.shape[2] >= c+1:
                    # RGB image
                    roi = image[y_min:y_max, x_min:x_max, c]
                else:
                    continue  # Skip if channel is out of bounds
            else:
                continue  # Unsupported image dimensions

            yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
            dist_from_center = np.sqrt((xx - (cx - x_min))**2 + (yy - (cy - y_min))**2)
            mask = dist_from_center <= radius
            values.extend(roi[mask].flatten())

        return np.median(values) if values else 0.0

def remove_blemish(image, x, y, radius, feather, opacity, channels_to_process):
    """
    Perform per-pixel blemish removal by sampling from surrounding circles.
    Handles edge cases where correction circles may extend beyond image boundaries.
    """
    corrected_image = image.copy()
    h, w = image.shape[:2]

    # Define angles for surrounding circles
    angles = [0, 60, 120, 180, 240, 300]
    surrounding_centers = []
    for angle in angles:
        rad = math.radians(angle)
        dx = int(math.cos(rad) * (radius * 1.5))  # 1.5 times the radius away
        dy = int(math.sin(rad) * (radius * 1.5))
        surrounding_centers.append((x + dx, y + dy))

    # Calculate medians for each surrounding circle and the target circle
    target_median = calculate_median_circle(image, x, y, radius, channels_to_process)
    surrounding_medians = [
        calculate_median_circle(image, cx, cy, radius, channels_to_process)
        for cx, cy in surrounding_centers
    ]

    # Determine the three correction circles closest to the target median
    median_diffs = [abs(median - target_median) for median in surrounding_medians]
    closest_indices = np.argsort(median_diffs)[:3]  # Indices of the three closest circles
    selected_circles = [surrounding_centers[i] for i in closest_indices]

    # Iterate through each channel
    for c in channels_to_process:
        # Iterate through each pixel in the target blemish circle
        for i in range(max(y - radius, 0), min(y + radius + 1, h)):
            for j in range(max(x - radius, 0), min(x + radius + 1, w)):
                dist = math.sqrt((j - x) ** 2 + (i - y) ** 2)
                if dist <= radius:
                    # Apply feathering based on distance
                    if feather > 0:
                        weight = max(0, min(1, (radius - dist) / (radius * feather)))
                    else:
                        weight = 1

                    # Collect corresponding pixel values from the selected correction circles
                    sampled_values = []
                    for (cx, cy) in selected_circles:
                        # Find the corresponding pixel position
                        corresponding_j = j + (cx - x)
                        corresponding_i = i + (cy - y)

                        # Ensure the corresponding pixel is within image bounds
                        if 0 <= corresponding_i < h and 0 <= corresponding_j < w:
                            if image.ndim == 2:
                                sampled_values.append(image[corresponding_i, corresponding_j])
                            elif image.ndim == 3:
                                if image.shape[2] == 1:
                                    sampled_values.append(image[corresponding_i, corresponding_j, 0])
                                elif image.shape[2] > c:
                                    sampled_values.append(image[corresponding_i, corresponding_j, c])
                                else:
                                    continue  # Skip if channel is out of bounds

                    if sampled_values:
                        # Calculate the median of the sampled values
                        median_val = np.median(sampled_values)
                    else:
                        # If no valid sampled pixels, retain the original pixel value
                        if image.ndim == 2:
                            median_val = image[i, j]
                        elif image.ndim == 3 and image.shape[2] ==1:
                            median_val = image[i, j,0]
                        else:
                            median_val = image[i,j,c]

                    # Blend the median value into the target pixel using opacity and feathering
                    if image.ndim ==2:
                        original_val = image[i, j]
                        blended_val = (1 - opacity * weight) * original_val + (opacity * weight) * median_val
                        corrected_image[i, j] = blended_val
                    elif image.ndim ==3 and image.shape[2] ==1:
                        original_val = image[i, j,0]
                        blended_val = (1 - opacity * weight) * original_val + (opacity * weight) * median_val
                        corrected_image[i, j,0] = blended_val
                    elif image.ndim ==3 and image.shape[2] >c:
                        original_val = image[i, j, c]
                        blended_val = (1 - opacity * weight) * original_val + (opacity * weight) * median_val
                        corrected_image[i, j, c] = blended_val

    return corrected_image

def float32_to_uint8(float_img):
    """
    Convert a float32 image to uint8 format.
    
    Parameters:
    - float_img: np.float32 array, input image
    
    Returns:
    - np.uint8 array, output image with values in range 0-255
    """
    # Check the current range of the image
    min_val = float_img.min()
    max_val = float_img.max()
    
    # Method 1: Simple scaling (for images already in 0-1 range)
    if 0 <= min_val and max_val <= 1:
        uint8_img = (float_img * 255).clip(0, 255).astype(np.uint8)

        
    
    # Method 2: Full normalization (for images with arbitrary ranges)
    else:
        # Normalize to 0-1 range first
        normalized_img = (float_img - min_val) / (max_val - min_val)
        # Then scale to 0-255
        uint8_img = (normalized_img * 255).clip(0, 255).astype(np.uint8)
    
    return uint8_img


def use_blob_detector(name, original_image, params, smoothing_params = None):
    """
    Use a blob detector to find and mask stars in an image.
    
    Parameters:
    - name: str, name of the image for saving output
    - original_image: np.array, the original image
    - params: cv2.SimpleBlobDetector_Params, parameters for the blob detector
    - smooth: int, smoothing factor for Gaussian blur
    
    Returns:
    - mask: np.array, binary mask of detected stars
    """
    
    # Convert to grayscale if not already
    if len(original_image.shape) == 3:
        img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = original_image

    # Apply Gaussian blur to reduce noise
    if smoothing_params is not None:
        blur = cv2.GaussianBlur(img_gray, (smoothing_params["kernel_size"], smoothing_params["kernel_size"]), smoothing_params["sigma"])
    else:
        blur = img_gray

    # Create the blob detector with the specified parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs (stars)
    keypoints = detector.detect(blur)

    img_con_punti = cv2.drawKeypoints(
    original_image, keypoints, np.array([]), (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
    
    # cv2.imwrite(f'./output_pipeline_luca/{name}_blob_keypoints.png', img_con_punti)

    # Create a mask for the detected blobs
    mask = np.zeros(img_gray.shape, dtype=np.uint8)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])       # coordinates of the center
        r = int(kp.size / 2)                      # estimated radius of the blob
        cv2.circle(mask, (x, y), r, 255, thickness=-1)  # draw blob in the mask

    # cv2.imwrite(f'./output_pipeline_luca/{name}_mask.png', mask)

    return mask, keypoints




def pipeline_blob_detector(original_image, config_params):
    # Imposta i parametri del detector
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = config_params["min_threshold"]
    params.maxThreshold = config_params["max_threshold"]
    params.thresholdStep = config_params["threshold_step"]

    # Filtro per area (modifica questi se rileva troppo)
    params.filterByArea = config_params["filter_by_area"]
    params.minArea = config_params["min_area"] 
    params.maxArea = config_params["max_area"]

    # Filtro per circolarità
    params.filterByCircularity = config_params["filter_by_circularity"]
    params.minCircularity = config_params["min_circularity"]  # più vicino a 1 = più circolare

    params.filterByConvexity = config_params["filter_by_convexity"]
    params.minConvexity = config_params["min_convexity"]  # più vicino a 1 = più convesso
    params.filterByInertia = config_params["filter_by_inertia"]
    params.minInertiaRatio = config_params["min_inertia_ratio"]  # più vicino a 1 = più simile a un cerchio

    # Cerca solo blob chiari su sfondo scuro
    params.filterByColor = config_params["filter_by_color"]
    params.blobColor = config_params["blob_color"]  # 0 = scuro, 255 = chiaro
    if config_params["smoothing"]:
        smoothing_params = config_params["smoothing_params"]
    else:
        smoothing_params = None

    mask, kp = use_blob_detector('m1_xs', original_image, params, smoothing_params)
    return mask, kp






















def main():
    # From the command line we can specify the config.file
    parsers = argparse.ArgumentParser()
    parsers.add_argument("-i", "--image", type=str)
    parsers.add_argument("-n", "--name", type=str)
    parsers.add_argument("-c", "--config", type=str, default="./config_pipeline.json")
    parsers.add_argument("-o", "--output", type=str, default="./output_blob")
    parsers.add_argument("-in", "--inpainting", type=str, default="false")
    args = parsers.parse_args()

    CONFIG_PATH = args.config
    NAME = args.name
    IMAGE_PATH = args.image
    OUTPUT_PATH = args.output
    INPAINTING = args.inpainting.lower() == "true"
    print(
        f"NAME = {NAME}\n"
        f"CONFIG_PATH = {CONFIG_PATH}\n"
        f"IMAGE_PATH = {IMAGE_PATH}\n"
        f"OUTPUT_PATH = {OUTPUT_PATH}\n"
        f"INPAINTING = {INPAINTING}\n"
    )
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    raw_img = fits.getdata(IMAGE_PATH)
    stretched_img = apply_stretch(raw_img)
    stretched_img = float32_to_uint8(stretched_img)

    cv2.imwrite(f"{OUTPUT_PATH}/stretched/{NAME}.png", stretched_img)

    star_dims = config["blob_detector"]["star_dimensions"]

    # Loop su ogni dimensione
    masks = {}
    kps = {}
    for dimension_name, config_params in star_dims.items():
        print(f"Working on {dimension_name} stars...")
        mask, kp = pipeline_blob_detector(stretched_img, config_params)
        masks[dimension_name] = mask
        kps[dimension_name] = kp
        print(f"Found {len(kp)} keypoints for {dimension_name} stars.")
    
    star_dims = config["mask_dilation"]["star_dimensions"]
    merged_mask = np.zeros_like(masks["small"])
    for dimension_name, config_params in star_dims.items():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config_params["kernel_size"], config_params["kernel_size"]))
        mask = cv2.dilate(masks[dimension_name], kernel, iterations=config_params["iterations"])
        merged_mask = cv2.bitwise_or(merged_mask, mask)
    cv2.imwrite(f"{OUTPUT_PATH}/masks/{NAME}.png", merged_mask)

    if INPAINTING:
        print("Inpainting...")
        inpainted = stretched_img.copy()
        star_dims = config["inpainting"]["star_dimensions"]
        for dimension_name, config_params in star_dims.items():
            add = config_params["add"]
            for kp in kps[dimension_name]:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                r = int(kp.size / 2) + add
                inpainted = remove_blemish(inpainted, x, y, r, feather=config_params["feather"], opacity=config_params["opacity"], channels_to_process=[0])
        cv2.imwrite(f"{OUTPUT_PATH}/inpainted/{NAME}.png", inpainted)
    
    


    





if __name__ == "__main__":
    main()