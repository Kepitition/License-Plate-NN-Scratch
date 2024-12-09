from NN_Scratch import *
from PIL import Image
import matplotlib.pyplot as plt
from crop_plate import contour_image, prepare_image_for_model, class_to_ascii

W1, b1, W2, b2, W3, b3 = load_model_parameters(r"Models/model_parameters.pkl")

contours = contour_image(r"Images/cropped/cropped_black_white_92_34WRP075.jpg")

for letter in contours:
    normalized_array, normalized_image = prepare_image_for_model(letter)

    plt.gray()
    plt.imshow(normalized_image, interpolation='nearest')
    plt.show()

    prediction = make_predictions(normalized_array,W1, b1, W2, b2, W3, b3)
    print(prediction[0],class_to_ascii(prediction[0]),prediction)
    #print(class_to_ascii(prediction[0]))


# image_path = r"Images/processed_images/cropped_plate_83_SEYLSS.jpg_char_20.png"
# image = Image.open(image_path)
# image_array = np.array(image)
# image_array = np.where(image_array == 0, 255, 0)
# #image_array = image_array.T
# normalized_array = image_array / 255.
# normalized_array = normalized_array.flatten()[:, None]
# print(normalized_array)
# normalized_image = normalized_array.reshape((28, 28)) * 255
#
# plt.gray()
# plt.imshow(normalized_image, interpolation='nearest')
# plt.show()
#
# W1, b1, W2, b2, W3, b3 = load_model_parameters(r"Models/model_parameters.pkl")
# prediction = make_predictions(normalized_array,W1, b1, W2, b2, W3, b3)
# print(prediction[0])