import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def PowerTransformation(grayImage):
    gamma = 7
    gamma_corrected = np.array(255 * (grayImage / 255) ** gamma, dtype='uint8')
    inverted_img = cv2.bitwise_not(gamma_corrected)
    blurr_img = cv2.blur(inverted_img, (1, 1))
    return blurr_img


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 1)
    return image


def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def blur_sides(image, left_ratio=0.1, right_ratio=0.1):
    height, width = image.shape[:2]
    left_width = int(width * left_ratio)
    right_width = int(width * right_ratio)

    # Create mask for left side blur
    left_mask = np.zeros((height, width), dtype=np.uint8)
    left_mask[:, :left_width] = 255

    # Create mask for right side blur
    right_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask[:, -right_width:] = 255

    # Apply blur to left and right side using masks
    left_blur = cv2.blur(image, (15, 15))
    right_blur = cv2.blur(image, (15, 15))
    left_blur = cv2.bitwise_and(left_blur, left_blur, mask=left_mask)
    right_blur = cv2.bitwise_and(right_blur, right_blur, mask=right_mask)

    # Combine original image with blurred sides
    blurred_image = cv2.addWeighted(image, 1, left_blur, 0.5, 0)
    blurred_image = cv2.addWeighted(blurred_image, 1, right_blur, 0.5, 0)

    return blurred_image


def perform_ocr(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image,(256,128))
    plt.subplot(3, 3, 1)
    plt.imshow(gray_image)
    plt.title('Gray Image')

    # Perform preprocessing steps as required
    gamma_transformed_image = PowerTransformation(gray_image)
    plt.subplot(3, 3, 2)
    plt.imshow(gamma_transformed_image)
    plt.title('Gamma')

    no_noise = noise_removal(gamma_transformed_image)
    plt.subplot(3, 3, 3)
    plt.imshow(no_noise)
    plt.title('No noise')

    # Blur sides of the image
    blurred_image = blur_sides(no_noise, left_ratio=0.3, right_ratio=0.3)
    plt.subplot(3, 3, 4)
    plt.imshow(blurred_image)
    plt.title('Blurred Sides')

    dilated_image = thick_font(blurred_image)
    plt.subplot(3, 3, 5)
    plt.imshow(dilated_image)
    plt.title('Dilated image')

    eroded_image = thin_font(dilated_image)
    plt.subplot(3, 3, 6)
    plt.imshow(eroded_image)
    plt.title('Eroded Image')

    # Perform OCR
    custom_config = r'-l eng --oem 3 --psm 10 tessedit_char_whitelist=0123456789'
    ocr_result = pytesseract.image_to_string(eroded_image, config=custom_config)
    ocr_result = ocr_result.replace("\\", "").replace(" ", "").replace("o", "0").replace("O", "0").replace(",","").replace("i", "1")
    return ocr_result.strip()


image = cv2.imread('Images/frame_925.0.jpg')
text = perform_ocr(image)
print(text)
plt.show()
