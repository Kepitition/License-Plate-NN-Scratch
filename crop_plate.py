import cv2
import numpy as np

# License plate should be uploaded as black and white images.
def contour_image(full_image, threshold_value = 100):
    emptylist = []
    image = cv2.imread(full_image, cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    processed_image = cv2.erode(dilated_image, kernel, iterations=1)

    if image is None:
        print(f"Skipping {full_image} as it's not a valid image.")

    #_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    _, binary_image = cv2.threshold(processed_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for idx, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small noise
        if 5 < w < 150 and 10 < h < 200:
            # Crop the character and resize to 28x28
            character = binary_image[y:y + h, x:x + w]
            resized_character = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)
            _, binary_resized_character = cv2.threshold(resized_character, 127, 255, cv2.THRESH_BINARY)
            emptylist.append(binary_resized_character)

    return emptylist

# Returns image for model ready and ready for print
def prepare_image_for_model(letter):
    letter_array = np.array(letter)
    letter_array1 = np.where(letter_array == 0, 255, 0)
    #letter_array2 = letter_array1.T
    normalized_array = (letter_array1 / 255.).flatten()[:, None]
    #normalized_array = normalized_array.flatten()[:, None]
    normalized_image = normalized_array.reshape((28, 28)) * 255

    return normalized_array, normalized_image

def class_to_ascii(prediction_index):
    class_to_ascii = {
        0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57,
        10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74,
        20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84,
        30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101,
        40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116
    }

    return chr(class_to_ascii.get(prediction_index, None))

