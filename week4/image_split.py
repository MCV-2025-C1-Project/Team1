import cv2
import numpy as np

def split_images(img, debug=False):
    """
    Split an image into two artworks if they are separated horizontally, otherwise return the original.
    
    Args:
        img: Input image (BGR format)
        debug: If True, display the morphological gradient after opening
    
    Returns:
        tuple: (success_flag, result)
            - If split: (True, (left_image, right_image))
            - If not split: (False, original_image)
    """
    # Step 1: Compute morphological gradient of S and V components
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    # Compute morphological gradients
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    grad_s = cv2.morphologyEx(s_channel, cv2.MORPH_GRADIENT, kernel_grad)
    grad_v = cv2.morphologyEx(v_channel, cv2.MORPH_GRADIENT, kernel_grad)
    
    # Combine gradients
    gradient_magnitude = cv2.add(grad_s, grad_v)
    
    # Threshold: Remove gradient values below 10% of max, set rest to 1 (binary mask)
    max_value = gradient_magnitude.max()
    threshold_value = 0.1 * max_value
    binary = np.where(gradient_magnitude >= threshold_value, 1, 0).astype(np.uint8)
    
    # Step 2: Morphological opening with 10x10 square structure element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    opened = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    # Sort components by area (excluding background which is label 0)
    if num_labels < 3:  # Less than 2 components (background + 1 component)
        return False, img
    
    # Get areas and sort (excluding background at index 0)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    if len(areas) < 2:
        return False, img
    
    # Get the two largest components
    label1, area1 = areas[0]
    label2, area2 = areas[1]
    
    # Filter to keep only the two largest components
    filtered_mask = np.zeros_like(opened)
    filtered_mask[labels == label1] = 1
    filtered_mask[labels == label2] = 1
    
    # Get bounding boxes
    x1, y1, w1, h1 = stats[label1, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
    x2, y2, w2, h2 = stats[label2, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
    
    # Debug: Display morphological gradient after opening and filtering
    if debug:
        import matplotlib.pyplot as plt
        print(f"Max gradient value: {max_value}")
        print(f"Threshold (10% of max): {threshold_value}")
        print(f"Number of connected components (including background): {num_labels}")
        print(f"Number of foreground components: {num_labels - 1}")
        print(f"Component 1 - Area: {area1}, BBox: x={x1}, y={y1}, w={w1}, h={h1}")
        print(f"Component 2 - Area: {area2}, BBox: x={x2}, y={y2}, w={w2}, h={h2}")
        
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title('Morphological Gradient')
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('After Threshold (10%)')
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(opened, cmap='gray')
        plt.title('After Opening (10x10)')
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(filtered_mask, cmap='gray')
        plt.title('Two Largest Components')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Step 3: Check if rectangles are horizontally separated
    # One must be completely to the left of the other
    rect1 = (x1, x1 + w1)  # (left, right) edges
    rect2 = (x2, x2 + w2)
    
    # Check if they are horizontally separated (no horizontal overlap)
    horizontally_separated = (rect1[1] < rect2[0]) or (rect2[1] < rect1[0])
    
    if not horizontally_separated:
        # Rectangles are NOT horizontally separated, don't split
        if debug:
            print("Rectangles are NOT horizontally separated - NOT splitting")
        return False, img
    
    # Step 4: Rectangles are horizontally separated, split by middle distance
    # Determine which rectangle is on the left
    if x1 < x2:
        left_x, left_w = x1, w1
        right_x = x2
    else:
        left_x, left_w = x2, w2
        right_x = x1
    
    # Calculate split position (middle distance between rectangles)
    left_edge = left_x + left_w  # right edge of left rectangle
    right_edge = right_x  # left edge of right rectangle
    split_x = (left_edge + right_edge) // 2
    
    if debug:
        print(f"Rectangles ARE horizontally separated - SPLITTING at x={split_x}")
        print(f"Left edge: {left_edge}, Right edge: {right_edge}")
    
    # Split the image
    left_image = img[:, :split_x].copy()
    right_image = img[:, split_x:].copy()
    
    # Step 5: Verify split validity - check if either image is too small
    original_width = img.shape[1]
    left_width = left_image.shape[1]
    right_width = right_image.shape[1]
    
    min_width_threshold = 0.25 * original_width
    
    if left_width < min_width_threshold or right_width < min_width_threshold:
        if debug:
            print(f"Split rejected: One image is too small")
            print(f"Original width: {original_width}")
            print(f"Left width: {left_width} ({100*left_width/original_width:.1f}%)")
            print(f"Right width: {right_width} ({100*right_width/original_width:.1f}%)")
            print(f"Minimum threshold: {min_width_threshold:.1f} (25%)")
        return False, img
    
    if debug:
        print(f"Split accepted: Both images are large enough")
        print(f"Left width: {left_width} ({100*left_width/original_width:.1f}%)")
        print(f"Right width: {right_width} ({100*right_width/original_width:.1f}%)")
    
    return True, (left_image, right_image)