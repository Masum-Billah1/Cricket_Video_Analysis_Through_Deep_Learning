import cv2
import numpy as np
import os
import pytesseract
import asyncio
import aiofiles

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def PowerTransformation(grayImage):
    gamma = 7
    gamma_corrected = np.array(255*(grayImage / 255) ** gamma, dtype = 'uint8')
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
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def extract_score(image_path):
    img = cv2.imread(image_path)
    gray_image = grayscale(img)
    gamma_transformed_image = PowerTransformation(gray_image)
    no_noise = noise_removal(gamma_transformed_image)
    dilated_image = thick_font(no_noise)
    eroded_image = thin_font(dilated_image)

    custom_config = r'-l eng --oem 3 --psm 10 tessedit_char_whitelist=0123456789'
    ocr_result = pytesseract.image_to_string(eroded_image, config=custom_config)
    ocr_result = ocr_result.replace("\\", "").replace(" ", "").replace("o", "0").replace("O", "0").replace(",","")
    return ocr_result.strip()


def process_images_in_folder(folder_path):
    substr = "-"
    with open('scores.txt', 'w') as file:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                score = extract_score(file_path)
                if (score.__contains__(substr)):
                    file.write(f"Score from {file_name}: {score}\n")
                    # print(f"Score from {file_name}: {score}\n")



def main(folder_path):
    process_images_in_folder(folder_path)
