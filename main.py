""" This program detects and classifies numbers in a yatzy sheet  """
import argparse
import tensorflow as tf
import numpy as np

import cv2
import yatzy_sheet
import cv_utils


print('Reading tensorflow model...')
print(tf.__version__)
predict_model = tf.keras.models.load_model('./models/model_tensorflow')
predict_model.summary()


img_path = './assets/sample_sheets/sample_yatzy.jpg'
num_rows = 19  # specify num rows for the yatzy sheet, original yatzy has 19


parser = argparse.ArgumentParser()

parser.add_argument("--num_rows", help="set num rows of the yatzy grid")
parser.add_argument("--img_path", help="specify path to input image")
parser.add_argument("--debug", help="specify debug to stop at show_window")

args = parser.parse_args()

if args.num_rows:
    num_rows = int(args.num_rows)

if args.debug:
    cv_utils.set_debug(bool(args.debug))

if args.img_path:
    img_path = args.img_path

print("Reading image from path", img_path)
input_img = cv2.imread(img_path)

img_yatzy_sheet, img_binary_yatzy_sheet, img_binary_only_numbers, yatzy_cells_bounding_rects = yatzy_sheet.generate_yatzy_sheet(input_img, num_rows_in_grid=num_rows)

# Debugging step
img_yatzy_cells = img_yatzy_sheet.copy()
cv_utils.draw_bounding_rects(img_yatzy_cells, yatzy_cells_bounding_rects)
cv_utils.show_window('img_yatzy_cells', img_yatzy_cells)


digit_contours = cv_utils.get_external_contours(img_binary_only_numbers)

# Iterate over contours and predict numbers if the contours belong to a yatzy cell
for i, cnt in enumerate(digit_contours):

    digit_bounding_rect = cv2.boundingRect(cnt)
    x, y, w, h = digit_bounding_rect

    # Identify if and to which yatzy cell this bounding rect belongs to
    yatzy_cell = yatzy_sheet.validate_and_find_yatzy_cell(yatzy_cells_bounding_rects, digit_bounding_rect)
    if yatzy_cell is None:
        continue

    # Black/white binary version of the roi
    roi = img_binary_yatzy_sheet[y:y+h, x:x+w]

    roi_fit_20x20 = 20 / max(roi.shape[0], roi.shape[1])

    # Resize preserving binary format with INTER_NEAREST
    roi = cv2.resize(roi, None, fx=roi_fit_20x20, fy=roi_fit_20x20, interpolation=cv2.INTER_NEAREST)

    roi_background = np.zeros((28, 28), dtype=roi.dtype)
    # Place the digit in the roi_background, 4 from top and 4 from left.
    roi_background[4:4+roi.shape[0], 4:4+roi.shape[1]] = roi

    # Save the original roi
    cv2.imwrite("./assets/roi/original/roi_" +
                str(i) + ".png", roi_background)

    # Get the translation based on center of mass
    delta_x, delta_y = cv_utils.get_com_shift(roi_background)

    # Shift
    roi_background = cv_utils.shift_by(roi_background, delta_x, delta_y)
    cv2.imwrite("./assets/roi/shifted/roi_" +
                str(i) + ".png", roi_background)

    # Preprocess for prediction
    roi_background = roi_background - 127.5
    roi_background /= 127.5

    # Log loss probabilities from our softmax classifier
    prediction = predict_model(np.reshape(roi_background, (1, 28, 28, 1)))

    predicted_digit = np.argmax(prediction)

    # Mark them on the image
    cv2.rectangle(img_yatzy_sheet, (x, y),
                  (x+w, y+h), (100, 10, 100), 1)

    cv2.putText(img_yatzy_sheet, str(predicted_digit), (x + int(w/2), y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)


cv_utils.show_window('img_yatzy_sheet', img_yatzy_sheet, debug=True)
cv_utils.show_window('img_binary_yatzy_sheet', img_binary_yatzy_sheet)
cv_utils.show_window('img_binary_only_numbers', img_binary_only_numbers)
