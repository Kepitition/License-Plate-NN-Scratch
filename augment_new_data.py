import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# Define dataset path and output folder
dataset_path = r"Images/cropped"
output_folder = r"Images/processed_images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each image in the dataset path
for filename in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, filename)

    # Load the image in grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Skipping {filename} as it's not a valid image.")
        continue

    # Threshold the image
    threshold_value = 100
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Process each contour
    for idx, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small noise
        if w > 5 and h > 5:
            # Crop the character and resize to 28x28
            character = binary_image[y:y + h, x:x + w]
            resized_character = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)
            _, binary_resized_character = cv2.threshold(resized_character, 127, 255, cv2.THRESH_BINARY)

            # Save the resized character to the output folder
            output_filename = os.path.join(output_folder, f"{filename}_char_{idx + 1}.png")
            cv2.imwrite(output_filename, binary_resized_character)
            print(f"Saved: {output_filename}")

print("Processing complete.")
