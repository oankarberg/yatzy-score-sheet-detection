from functools import cmp_to_key
import cv2
import numpy as np
import cv_utils


MIN_CELL_DIGIT_HEIGHT_RATIO = 2.5


def __should_merge_lines(line_a, line_b, rho_distance, theta_distance):
    rho_a, theta_a = line_a[0].copy()
    rho_b, theta_b = line_b[0].copy()
    if(rho_b == rho_a and theta_b == theta_b):
        return False

    # Use degree for more intuitive user format
    theta_b = int(180 * theta_b / np.pi)
    theta_a = int(180 * theta_a / np.pi)

    # In Q3 or Q4, See merge_lines method
    if rho_b < 0:
        theta_b = theta_b - 180

    # In Q3 or Q4, See merge_lines method
    if rho_a < 0:
        theta_a = theta_a - 180

    rho_a = np.abs(rho_a)
    rho_b = np.abs(rho_b)

    diff_theta = np.abs(theta_a - theta_b)
    rho_diff = np.abs(rho_a - rho_b)

    if(rho_diff < rho_distance and diff_theta < theta_distance):
        return True

    return False


def resize_to_right_ratio(img, interpolation=cv2.INTER_LINEAR, width=750):
    """ Set ratio to resize to width """
    ratio_width = width / img.shape[1]
    # Resize
    return cv2.resize(img, None, fx=ratio_width, fy=ratio_width, interpolation=interpolation)


def merge_lines(line_a, line_b):
    """ Merge line_a and line_b by the average rho and angle.

            |
       Q3   |   Q4
            |
    -----------------
            | 
       Q2   |   Q1
            |

        Rho and theta works in the clockwise direction starting from Q1.  If rho < 0, lines
        are drawn in Q3 or Q4, else in Q1, Q2.  Theta is always positive going from 0 to PI
        We have to consider this when merging the lines
    """
    rho_b, theta_b = line_b[0]
    rho_a, theta_a = line_a[0]

    # We are in Q3 or Q4 in unit circle
    # Shift theta to be negative (inside Q3 to Q4)
    if rho_b < 0:
        rho_b = np.abs(rho_b)
        theta_b = theta_b - np.pi

    # We are in Q3 or Q4 in unit circle,
    # Shift theta to be negative (inside Q3 to Q4)
    if rho_a < 0:
        rho_a = np.abs(rho_a)
        theta_a = theta_a - np.pi

    average_theta = (theta_a + theta_b) / 2
    average_rho = (rho_a + rho_b) / 2
    # We are in Q3 or Q4 after averaging both lines, format to OpenCV HoughLines format
    if average_theta < 0:
        # This rho is negative
        average_rho = -average_rho
        # Re-format to positive values for opencv HoughLines 0 to PI
        average_theta = np.abs(average_theta)

    return [[average_rho, average_theta]]


def get_merged_line(lines, line_a, rho_distance, degree_distance):
    """ Merges all line in lines with the distance to line_a iteratively.

    Returns:
        list: a list of estimated lines
    """
    for i, line_b in enumerate(lines):
        if line_b is False:
            continue
        if __should_merge_lines(line_a, line_b, rho_distance, degree_distance):
            # Update line A on every iteration
            line_a = merge_lines(line_a, line_b)
            # Don't use B again
            lines[i] = False

    return line_a


def get_adaptive_binary_image(img):
    """ Returns a binary image with adaptive threshold.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter for smoothing effect of shadows etc. before adaptive threshold
    # We use 5x5 kernel with Sigma = 0, which gives us StD = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # See https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#Mat%20getGaussianKernel(int%20ksize,%20double%20sigma,%20int%20ktype)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # We use Adaptive threshold to get threshold values based on different regions.
    # We use blocksize 7 for region size and 4 as constant to subtract from the Gaussian weighted sum
    # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    return cv2.adaptiveThreshold(gray, 255,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 4)


def merge_nearby_lines(lines, rho_distance=30, degree_distance=20):
    """ Merges nearby lines with the specified rho and degree distance.

    Args:
        lines (list): A list of lines (rho, theta), see OpenCV HoughLines
        rho_distance (int): Distance in rho for lines to merge (default is 30)
        degree_distacne (int): Distance in degrees for lines to merge (default is 20)

    Returns:
        list: a list of estimated lines
    """

    lines = lines if lines is not None else []
    estimated_lines = []
    for line in lines:
        if line is False:
            continue

        estimated_line = get_merged_line(
            lines, line, rho_distance, degree_distance)
        estimated_lines.append(estimated_line)

    return estimated_lines


def draw_lines(lines, img):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
            cv2.line(img, pt1, pt2,
                     (255, 255, 255), 2)


def sort_by_upper_left_pos(rect_a, rect_b):
    x_a, y_a, _, _ = rect_a
    x_b, y_b, w_b, _ = rect_b

    # These code of lines are for handling scewed grid,
    # but since we are only dealing with fully vertical/horizontal lines, this is not needed.
    # Good for future if we want to work with scewed lines
    x_b_offset_positive = x_b + w_b / 3
    x_b_offset_negative = x_b - w_b / 3
    is_same_column = x_a < x_b_offset_positive and x_a > x_b_offset_negative

    if is_same_column:
        return y_a - y_b
    return (x_a - x_b_offset_positive)


def get_rotated_yatzy_sheet(img, img_adaptive_binary):
    # Find the biggest outer contour to locate the Yatzy Sheet. We use RETR_EXTERNAL and discard nested contours.
    contours = cv_utils.get_external_contours(img_adaptive_binary)
    biggest_contour = cv_utils.get_biggest_intensity_contour(contours)

    img_raw_yatzy_sheet = cv_utils.get_rotated_image_from_contour(img, biggest_contour)

    img_raw_yatzy_sheet = resize_to_right_ratio(img_raw_yatzy_sheet)
    img_binary_sheet_rotated = get_adaptive_binary_image(img_raw_yatzy_sheet)

    return img_raw_yatzy_sheet, img_binary_sheet_rotated


def generate_yatzy_sheet(img, num_rows_in_grid=19, max_num_cols=20):
    img = resize_to_right_ratio(img)
    img_adaptive_binary = get_adaptive_binary_image(img)

    cv_utils.show_window('img_adaptive_binary', img_adaptive_binary)

    # Find the biggest contour and rotate it
    img_yatzy_sheet, img_binary_yatzy_sheet = get_rotated_yatzy_sheet(img, img_adaptive_binary)

    # Get a painted grid with vertical / horizontal lines
    img_binary_grid, img_binary_only_numbers = get_yatzy_grid(img_binary_yatzy_sheet)

    # Get every yatzy grid cell as a sorted bounding rect in order to later locate numbers to correct cell
    yatzy_cells_bounding_rects, grid_bounding_rect = get_yatzy_cells_bounding_rects(img_binary_grid, num_rows_in_grid, max_num_cols)

    # Get the area of the yatzy grid from different versions of raw img
    img_binary_only_numbers = cv_utils.get_bounding_rect_content(img_binary_only_numbers, grid_bounding_rect)
    img_binary_yatzy_sheet = cv_utils.get_bounding_rect_content(img_binary_yatzy_sheet, grid_bounding_rect)
    img_yatzy_sheet = cv_utils.get_bounding_rect_content(img_yatzy_sheet, grid_bounding_rect)

    return img_yatzy_sheet, img_binary_yatzy_sheet, img_binary_only_numbers, yatzy_cells_bounding_rects


def get_yatzy_grid(img_binary_sheet):
    """ Returns a binary image with a grid and the input image containing only horizontal/vertical lines.

    Args:
        img_binary_sheet ((rows,col) array): binary image

    Returns:
        img_binary_grid: an image containing painted vertically and horizontally lines
        img_binary_sheet_only_digits: an image containing only(mostly) handwritten digits
    """
    height, width = img_binary_sheet.shape

    img_binary_sheet_morphed = img_binary_sheet.copy()

    # Now we have the binary image with adaptive threshold.
    # We need to do some morphylogy operations in order to strengthen thin lines, remove noise, and also handwritten stuff.
    # We only want the horizontal / vertical pixels left before we start identifying the grid. See http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm

    # CLOSING: (dilate -> erode) will fill in background (black) regions with White. Imagine sliding struct element
    # in the background pixel, if it cannot fit the background completely(touching the foreground), fill this pixel with white

    # OPENING: ALL FOREGROUND PIXELS(white) that can fit the structelement will be white, else black.
    # Erode -> Dilate

    # Erosion: If the structuring element can fit inside the forground pixel(white), then keep white, else set to black
    # Dilation: For every background pixel(black), if one of the foreground(white) pixels are present, set this background (black) to foreground.

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_binary_sheet_morphed = cv2.morphologyEx(img_binary_sheet_morphed, cv2.MORPH_DILATE, kernel)

    cv_utils.show_window('morph_dilate_binary_img', img_binary_sheet_morphed)

    sheet_binary_grid_horizontal = img_binary_sheet_morphed.copy()
    sheet_binary_grid_vertical = img_binary_sheet_morphed.copy()

    # We use relative length for the structuring line in order to be dynamic for multiple sizes of the sheet.
    structuring_line_size = int(width / 5.0)

    # Try to remove all vertical stuff in the image,
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_line_size, 1))
    sheet_binary_grid_horizontal = cv2.morphologyEx(sheet_binary_grid_horizontal, cv2.MORPH_OPEN, element)

    # Try to remove all horizontal stuff in image, Morph OPEN: Keep everything that fits structuring element i.e vertical lines
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, structuring_line_size))

    sheet_binary_grid_vertical = cv2.morphologyEx(sheet_binary_grid_vertical, cv2.MORPH_OPEN, element)

    # Concatenate the vertical/horizontal lines into grid
    img_binary_sheet_morphed = cv2.add(
        sheet_binary_grid_vertical, sheet_binary_grid_horizontal)

    cv_utils.show_window("morph_keep_only_horizontal_lines", sheet_binary_grid_horizontal)
    cv_utils.show_window("morph_keep_only_vertical_lines", sheet_binary_grid_vertical)
    cv_utils.show_window("concatenate_vertical_horizontal", img_binary_sheet_morphed)
    """
        Time to get a solid grid, from what we see  above, the grid is still not fully filled (sometimes)
        since the paper is not fully straight on the table etc. For this we use Hough Transform
        Hough transform identifies points (x,y) on the same line. 

    """
    # We ideally should choose np.pi / 2 for the Theta accumulator, since we only want lines in 90 degrees and 0 degrees.
    rho_accumulator = 1
    angle_accumulator = np.pi / 2
    # Min vote for defining a line
    threshold_accumulator_votes = int(width/2)

    # Find lines in the image according to the Hough Algorithm
    grid_lines = cv2.HoughLines(img_binary_sheet_morphed, rho_accumulator,
                                angle_accumulator, threshold_accumulator_votes)

    img_binary_grid = np.zeros(
        img_binary_sheet_morphed.shape, dtype=img_binary_sheet_morphed.dtype)

    # Since we can have multiple lines for same grid line, we merge nearby lines
    grid_lines = merge_nearby_lines(grid_lines)

    draw_lines(grid_lines, img_binary_grid)

    # Since all sheets does not have outerborders. We draw a rectangle around the
    outer_border = np.array([
        [1, height-1],  # Bottom Left
        [1, 1],  # Top Left
        [width-1, 1],  # Top Right
        [width-1, height-1]  # Bottom Right
    ])
    cv2.drawContours(img_binary_grid, [outer_border], 0, (255, 255, 255), 3)

    # Remove the grid from the binary image an keep only the digits.
    img_binary_sheet_only_digits = cv2.bitwise_and(img_binary_sheet, 255 - img_binary_sheet_morphed)

    cv_utils.show_window("yatzy_grid_binary_lines", img_binary_grid)

    return img_binary_grid, img_binary_sheet_only_digits


def __filter_by_dim(val, target_width, target_height):
    # Remove cells outside of target width/height
    offset_width = target_width * 0.3
    offset_height = target_height * 0.3
    _, _, w, h = val
    return target_width - offset_width < w < target_width + offset_width and target_height - offset_height < h < target_height + offset_height


def get_most_common_area(bounding_rects, cell_resolution):
    cell_areas = [int(w*h/cell_resolution) for _, _, w, h in bounding_rects]

    # 1-Dimensional groups
    counts = np.bincount(cell_areas)
    return bounding_rects[np.argmax(counts)]


def validate_and_find_yatzy_cell(cells, bounding_rect):
    """ Finds a corresponding cell for the bounding_rect. """
    roi_x, roi_y, roi_w, roi_h = bounding_rect
    roi_center_x = roi_x + int(roi_w/2)
    roi_center_y = roi_y + int(roi_h/2)
    _, _, cell_width, cell_height = cells[0]
    # Discard noise contours
    if(not 2 < roi_w < cell_width or not (cell_height/MIN_CELL_DIGIT_HEIGHT_RATIO) < roi_h < cell_height):
        return None

    found_cell = None
    for rect in cells:
        x, y, w, h = rect
        # Verify which cell roi_center belongs to
        if(x < roi_center_x < x + w and y < roi_center_y < y + h):
            found_cell = rect
            break

    return found_cell


def get_yatzy_cells_bounding_rects(img_binary_grid, num_rows_in_grid=19, max_num_cols=20):
    """ Returns a list with sorted bounding rect for every yatzy grid cell. """
    # Now we have the grid in img_binary_grid
    # Lets start identifying the cells by getting all contours from the vertical / horizontal binary img grid
    binary_grid_contours, _ = cv2.findContours(img_binary_grid, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

    sheet_width = img_binary_grid.shape[1]

    cell_min_width = (sheet_width/max_num_cols)

    # Clean grid from small contours
    yatzy_cells_bounding_rects = [cv2.boundingRect(cnt) for cnt in binary_grid_contours if cv_utils.wider_than(cnt, cell_min_width)]

    # Define resolution for cell area "bins"
    cell_resolution = (sheet_width/50) ** 2

    _, _, target_width, target_height = get_most_common_area(yatzy_cells_bounding_rects, cell_resolution)

    if len(yatzy_cells_bounding_rects) < num_rows_in_grid:
        print("ERROR: Not enough grid cells found.")

    yatzy_cells_bounding_rects = list(filter(lambda x: __filter_by_dim(x, target_width, target_height), yatzy_cells_bounding_rects))

    num_cells = len(yatzy_cells_bounding_rects)
    correct_num_cells_in_grid = (num_cells >= num_rows_in_grid and num_cells % num_rows_in_grid == 0)

    if not correct_num_cells_in_grid:
        print("ERROR: not correct number fo cells found in grid, num found:", num_cells)

    yatzy_grid_bounding_rect = cv_utils.concatenate_bounding_rects(yatzy_cells_bounding_rects)

    shift_x, shift_y, _, _ = yatzy_grid_bounding_rect
    yatzy_cells_bounding_rects = list(map(lambda x: cv_utils.move_bounding_rect(x, -shift_x, -shift_y), yatzy_cells_bounding_rects))
    yatzy_cells_bounding_rects = sorted(
        yatzy_cells_bounding_rects, key=cmp_to_key(sort_by_upper_left_pos))
    return yatzy_cells_bounding_rects, yatzy_grid_bounding_rect
