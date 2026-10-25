import cv2
import numpy as np
import os

def identify_and_mask_ignoring_shadows(folder_path):
    """
    1. ignores shadows by setting a high brightness floor (V >= 80).
    2. identifies the dominant color first using HSV ranges.
    3. evaluates shape second (printing validation status).
    4. unconditionally highlights the matched color region in royal blue and saves it.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # create an output directory for the highlighted images
    output_dir = "output_results_no_shadows"
    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    print(f"Scanning folder: '{folder_path}' (Shadows Ignored, Color -> Shape -> Unconditional Blue Masking)...\n")

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(folder_path, filename)
        
        # load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image from {image_path}")
            continue

        # resize for consistent processing
        image = cv2.resize(image, (400, 400))
        
        #convert BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # ==========================================
        # define the HSV categories of colors
        # ==========================================
        lower_red1 = np.array([0, 70, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 80])
        upper_red2 = np.array([180, 255, 255])
        
        lower_yellow = np.array([10, 70, 80])
        upper_yellow = np.array([32, 255, 255])
        
        lower_granny = np.array([30, 30, 80])
        upper_granny = np.array([75, 255, 255])
        
        # Generate Masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        green_mask = cv2.inRange(hsv, lower_granny, upper_granny)

        # ==========================================
        # step 1: identify the color of the object
        # ==========================================
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        color_counts = {
            "Red": red_pixels,
            "Yellow": yellow_pixels,
            "Green": green_pixels
        }

        dominant_color = max(color_counts, key=color_counts.get)
        max_pixels = color_counts[dominant_color]

        # select the active mask for the winning color
        if dominant_color == "Red":
            active_mask = red_mask
        elif dominant_color == "Yellow":
            active_mask = yellow_mask
        else:
            active_mask = green_mask

        if max_pixels < 500:
            print(f"-> File '{filename}': No significant color match found (ignoring shadows).")
        else:
            print(f"-> File '{filename}': Identified color as **{dominant_color}** (shadows ignored).")

        # ==========================================
        # step 2: check to see if the shape resembles an apple (circular)
        # ==========================================
        contours, _ = cv2.findContours(active_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        meets_shape_requirement = False
        shape_status = "Failed shape criteria (not an apple shape)"

        if contours and max_pixels >= 500:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if area > 500 and perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if circularity > 0.5 and solidity > 0.8:
                    meets_shape_requirement = True
                    shape_status = "Meets shape criteria (Validated Apple shape!)"

        print(f"   [Shape Status]: {shape_status}")

        # ==========================================
        # step 3: add an overlay of blue on the detected primary color
        # ==========================================
        # pretty blue in OpenCV BGR format: Blue=225, Green=105, Red=65
        royal_blue = np.array([225, 105, 65], dtype=np.uint8)

        overlay = image.copy()
        overlay[active_mask > 0] = royal_blue
        
        # blend overlay (60% blue tint, 40% original image)
        highlighted_image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        #save the result
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, highlighted_image)

    print(f"\nProcessing complete! Results saved in folder: '{output_dir}/'")

if __name__ == "__main__":
    target_folder = "dataset"
    identify_and_mask_ignoring_shadows(target_folder)