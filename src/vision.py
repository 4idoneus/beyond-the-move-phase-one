import cv2
import numpy as np

def process_image_to_board(image_path, debug=False):
    """
    Main entry point.
    Returns:
        - board_state: 19x19 numpy array (0=Empty, 1=Black, 2=White)
        - debug_image: Image with the grid drawn on it (for verification)
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image from {image_path}")

    # 1. Detect Board & Warp
    corners = find_board_corners(original)
    warped_board = four_point_transform(original, corners)

    # 2. Analyze Grid
    # margin_ratio=0.04 accounts for the wooden edge of the board
    board_state, debug_image = analyze_grid(warped_board, margin_ratio=0.06)
    
    return board_state, debug_image

def find_board_corners(image):
    """ Finds the largest square-ish contour. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Auto-canny parameter
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blurred, lower, upper)
    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    display_cnt = None
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Check if 4 points and reasonable size
        if len(approx) == 4 and cv2.contourArea(c) > (image.shape[0] * image.shape[1] * 0.1):
            display_cnt = approx
            break
            
    if display_cnt is None:
        # Fallback: Use full image
        h, w = image.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    
    return display_cnt.reshape(4, 2)

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def analyze_grid(warped_image, margin_ratio=0.0):
    """
    Analyzes the grid.
    margin_ratio: How much of the edge is 'wood' before the first line starts.
    """
    debug_img = warped_image.copy()
    
    output_size = 800
    warped = cv2.resize(warped_image, (output_size, output_size))
    debug_img = cv2.resize(debug_img, (output_size, output_size))
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Calculate grid step
    total_grid_width = output_size * (1.0 - (2 * margin_ratio))
    step = total_grid_width / 18.0
    start_offset = output_size * margin_ratio
    
    board_arr = np.zeros((19, 19), dtype=int)
    radius = int(step * 0.3)

    # --- DYNAMIC THRESHOLDING FIX ---
    # Calculate the median brightness of the board (proxy for wood color)
    median_bg = np.median(gray)
    
    # Define thresholds relative to the background wood color
    # Adjusted Sensitivities: 
    # Reduced the delta to 30 (was 50) to catch white stones that are not super bright.
    thresh_black = max(0, median_bg - 40)  # Keep black stricter to avoid shadows
    thresh_white = min(255, median_bg + 30) # Relax white to catch more stones

    for row in range(19):
        for col in range(19):
            x = int(start_offset + (col * step))
            y = int(start_offset + (row * step))
            
            x = max(radius, min(output_size - radius, x))
            y = max(radius, min(output_size - radius, y))
            
            # Debug Draw
            cv2.circle(debug_img, (x, y), 3, (0, 0, 255), -1)
            if col < 18: cv2.line(debug_img, (x, y), (int(x+step), y), (0, 255, 0), 1)
            if row < 18: cv2.line(debug_img, (x, y), (x, int(y+step)), (0, 255, 0), 1)
            
            roi = gray[y-radius:y+radius, x-radius:x+radius]
            if roi.size == 0: continue
            
            mean_val = np.mean(roi)
            
            # Use Dynamic Thresholds
            if mean_val < thresh_black:
                board_arr[row][col] = 1 # Black
                cv2.circle(debug_img, (x, y), radius, (255, 0, 0), 2)
            elif mean_val > thresh_white:
                board_arr[row][col] = 2 # White
                cv2.circle(debug_img, (x, y), radius, (0, 0, 255), 2)
    
    # --- ORIENTATION FIX ---
    # The image is processed Top-to-Bottom.
    # In standard Go coordinates (and the app), Row 0 is the Bottom (A1).
    # So we need to flip the array vertically so the Top of the image becomes the Top of the board matrix.
    board_arr = np.flipud(board_arr)
                
    return board_arr, debug_img