import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
# Formatting dataset images to csv file.
# Which works as this topology

# |-- Images (letters dataset)
# |    |-- Class 1
# |    |    |-- Class specific images
# |    +-- Class 2
# |    |     |-- Class specific images
# |    |     |...

## Define the path to the dataset and the output CSV file
dataset_path = r"Images/letters_dataset"
output_csv = "license_plate_data.csv"

# Initialize a list to hold all rows
data_rows = []

# Iterate through folders (labels)
for label in os.listdir(dataset_path):
    print(label)
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    # Iterate through each image in the folder
    for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)

        # Read the image and resize to 28x28
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        resized_image = cv2.resize(image, (28, 28))

        # Binarize the image
        _, binary_image = cv2.threshold(resized_image, 120, 255, cv2.THRESH_BINARY)

        # Flatten the image to a 1D array
        flattened_image = binary_image.flatten()

        # Prepend the label to the array
        row = [label] + flattened_image.tolist()

        # Append the row to the data list
        data_rows.append(row)

# Write the data to a CSV file
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_rows)

print(f"Data successfully saved to {output_csv}")


# Test the data by visualizing some samples
def test_data(csv_file, num_samples=5):
    print("\nTesting data conversion...")

    # Read back the data from the CSV
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    # Randomly select a few samples
    for i in range(num_samples):
        row = data[i]
        label = row[0]  # First column is the label
        pixels = np.array(row[1:], dtype=np.uint8)  # Remaining are pixel values

        # Reshape to 28x28 for visualization
        image = pixels.reshape((28, 28))

        # Display the binary image
        plt.title(f"Label: {label}")
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.show()
