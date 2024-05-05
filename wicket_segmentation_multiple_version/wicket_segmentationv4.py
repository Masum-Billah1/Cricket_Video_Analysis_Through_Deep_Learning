import cv2
import os
import random
import pytesseract
import numpy as np
import time
import re
from moviepy.video.io.VideoFileClip import VideoFileClip


def find_wicket_number(text,score_position,separator):
    """
    Find the digit after the hyphen in a given text.

    Args:
        text (str): Input text to search for a hyphen and extract digit after it.

    Returns:
        int: Digit after the hyphen if found, -1 otherwise.
    """
    if separator == "-":
        hyphen_index = text.find('-')  # Find the index of the first hyphen
    elif separator == "/":
        hyphen_index = text.find('/')
    text_length = (len(text))
    wicket = -1
    if hyphen_index != -1:
        # If a hyphen is found, assign the digit to wicket
        if score_position == "R" or score_position == "r":
            if (text_length > (hyphen_index+2)):
                if (text[hyphen_index+1] == '1' and text[hyphen_index+2] == '0'):
                    try:
                        wicket = int(10)
                    except ValueError:  # If after hyphen it is not an integer digit then assign -1 to wicket
                        wicket = -1
            else:
                try:
                    wicket = int(text[hyphen_index + 1])
                except ValueError: # If after hyphen it is not an integer digit then assign -1 to wicket
                    wicket = -1
        else:
            if (hyphen_index>1):
                if text[hyphen_index-2] == '1' and text[hyphen_index-1] == '0':
                    try:
                        wicket = int(10)
                    except ValueError:  # If after hyphen it is not an integer digit then assign -1 to wicket
                        wicket = -1
            else:
                try:
                    wicket = int(text[hyphen_index - 1])
                except ValueError:  # If after hyphen it is not an integer digit then assign -1 to wicket
                    wicket = -1
    # return wicket after hyphen
    return wicket

def is_last_wicket(ocr_result):
    pattern = r"^\d{2,3}$"   # 2/3 digit length
    print(f'10 wickets before, ocr_result is : {ocr_result}')
    match = re.match(pattern, ocr_result)  # Check if the pattern matches the OCR result
    return match is not None
def is_score(ocr_result,score_position,separator):
    """
    Check if the OCR result matches the score pattern.

    Args:
        ocr_result (str): Result obtained from OCR.

    Returns:
        bool: True if the pattern matches, False otherwise.
    """
    if score_position=="R" or score_position =="r":
        if separator == "-":
            pattern = r".*(\d{1,3})-(\d).*"
        else:
            pattern = r".*(\d{1,3})/(\d).*"
    else:
        if separator == "-":
            pattern = r"*(\d)-(\d{1,3}).*"
        else:
            pattern = r".*(\d)/(\d{1,3}).*"

    match = re.match(pattern, ocr_result)  # Check if the pattern matches the OCR result
    return match is not None


def grayscale(image):
    """
    Convert an image to grayscale.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Grayscale version of the input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def PowerTransformation(grayImage):
    """
    Apply power transformation to the input image.

    Args:
        grayImage (numpy.ndarray): Grayscale image.

    Returns:
        numpy.ndarray: Transformed image.
    """
    gamma = 7
    # Apply power transformation using gamma correction
    gamma_corrected = np.array(255 * (grayImage / 255) ** gamma, dtype='uint8')
    # Invert the image
    inverted_img = cv2.bitwise_not(gamma_corrected)
    # Blur the inverted image
    blurr_img = cv2.blur(inverted_img, (1, 1))
    return blurr_img


def noise_removal(image):
    """
    Remove noise from the input image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image after noise removal.
    """
    kernel = np.ones((1, 1), np.uint8)
    # Dilate the image to enhance features
    image = cv2.dilate(image, kernel, iterations=1)
    # Erode the image to remove noise
    image = cv2.erode(image, kernel, iterations=1)
    # Apply morphological closing to further remove noise
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # Apply median blur to smoothen the image
    image = cv2.medianBlur(image, 1)
    return image


def thick_font(image):
    """
    Apply thick font to the input image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with thick font applied.
    """
    # Invert the image
    image = cv2.bitwise_not(image)
    kernel = np.ones((1, 1), np.uint8)
    # Dilate the inverted image to make the text thicker
    image = cv2.dilate(image, kernel, iterations=1)
    # Invert the image again to get the original color
    image = cv2.bitwise_not(image)
    return image


def thin_font(image):
    """
    Apply thin font to the input image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with thin font applied.
    """
    # Invert the image
    image = cv2.bitwise_not(image)
    kernel = np.ones((1, 1), np.uint8)
    # Erode the inverted image to make the text thinner
    image = cv2.erode(image, kernel, iterations=1)
    # Invert the image again to get the original color
    image = cv2.bitwise_not(image)
    return image


def blur_sides(image, left_ratio=0.2, right_ratio=0.2):
    """
    Apply blur to the left and right sides of the input image.

    Args:
        image (numpy.ndarray): Input image.
        left_ratio (float): Ratio of width to blur on the left side.
        right_ratio (float): Ratio of width to blur on the right side.

    Returns:
        numpy.ndarray: Image with blurred sides.
    """
    height, width = image.shape[:2]
    left_width = int(width * left_ratio)
    right_width = int(width * right_ratio)

    # Create masks for left and right sides
    left_mask = np.zeros((height, width), dtype=np.uint8)
    left_mask[:, :left_width] = 255
    right_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask[:, -right_width:] = 255

    # Apply blur to left and right sides separately
    left_blur = cv2.blur(image, (15, 15))
    right_blur = cv2.blur(image, (15, 15))
    # Apply masks to retain the center and blur the sides
    left_blur = cv2.bitwise_and(left_blur, left_blur, mask=left_mask)
    right_blur = cv2.bitwise_and(right_blur, right_blur, mask=right_mask)

    # Combine the blurred sides with the original image
    blurred_image = cv2.addWeighted(image, 1, left_blur, 0.5, 0)
    blurred_image = cv2.addWeighted(blurred_image, 1, right_blur, 0.5, 0)

    return blurred_image


def get_video_duration(video_path):
    """
    Get the duration of the video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


def select_random_frame(video_path):
    """
    Select a random frame from the video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        numpy.ndarray: Randomly selected frame.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_selected = False
    frame = None

    while not frame_selected:
        random_frame_number = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
        ret, frame = cap.read()
        cv2.namedWindow("Random Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Random Frame", 1920, 1080)
        cv2.imshow("Random Frame", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            frame_selected = True
        else:
            print("Showing another random frame...")

    cap.release()
    cv2.destroyAllWindows()
    return frame


def crop_image(image):
    """
    Crop the image based on the selected ROI.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        tuple: Cropped image and ROI coordinates.
    """
    # Create a window to select Region of Interest (ROI)
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", 1920, 1080)
    # Select ROI using mouse drag
    roi = cv2.selectROI("Select ROI", image)
    # Crop the image using the selected ROI
    cropped_image = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    cv2.destroyAllWindows()
    return cropped_image, roi


def perform_ocr(image):
    """
    Perform Optical Character Recognition (OCR) on the input image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        str: Text recognized by OCR.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to a fixed size
    resized_image = cv2.resize(gray_image, (512, 128))
    # Apply power transformation to enhance contrast
    gamma_transformed_image = PowerTransformation(resized_image)
    # Remove noise from the image
    no_noise = noise_removal(gamma_transformed_image)
    # Apply thick font to make characters more distinguishable
    dilated_image = thick_font(no_noise)
    # Apply thin font to refine character shapes
    eroded_image = thin_font(dilated_image)
    # Perform OCR on the processed image
    custom_config = r'-l eng --oem 3 --psm 10 tessedit_char_whitelist=0123456789'
    ocr_result = pytesseract.image_to_string(eroded_image, config=custom_config)
    # Preprocess the OCR result to remove unwanted characters
    ocr_result = ocr_result.replace("\\", "").replace(" ", "").replace("o", "0").replace("O", "0").replace(",",
                                                                                                           "").replace(
        "i", "1")
    return ocr_result.strip()


def apply_crop_to_video(video_path, roi, extraction_interval,score_position,separator):
    """
    Apply the same crop to frames with specified interval in the video and extract the wicket from each frame.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to save cropped frames.
        roi (tuple): Coordinates of the Region of Interest (ROI).
        extraction_interval (int): Interval for frame extraction.

    Returns:
        list: List of tuples containing time and wicket information.
    """
    score = []
    cap = cv2.VideoCapture(video_path)

    # Create the cropped frames output folder if it doesn't exist
    crop_output_folder = "cropped_images"
    os.makedirs(crop_output_folder, exist_ok=True)

    # Initialize OCR result file
    ocr_result_file = open(os.path.join("ocr_results.txt"), "w")
    duration = int(get_video_duration(video_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(int(fps * extraction_interval), 1)
    wicket_in_previous_frame = 0

    for i in range(0, duration * fps, frame_interval):
        wicket_in_this_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            seconds = i / fps
            # Crop the frame using the selected ROI
            cropped_frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            # Perform OCR on the cropped frame
            ocr_result = perform_ocr(cropped_frame)

            # Save the cropped frame
            output_path = os.path.join(crop_output_folder, f"frame_{seconds:.1f}.jpg")
            cv2.imwrite(output_path, cropped_frame)

            if is_score(ocr_result,score_position,separator):
                # Extract wicket information from OCR result
                wicket_in_this_frame = int(find_wicket_number(ocr_result, score_position,separator))

                if (wicket_in_this_frame - wicket_in_previous_frame == 1) or (
                        wicket_in_this_frame - wicket_in_previous_frame == 0) and wicket_in_this_frame != -1:
                    score.append((seconds, wicket_in_this_frame))
                    ocr_result_file.write(f"Frame {seconds:.1f}: {wicket_in_this_frame}\n")
                else:
                    wicket_in_this_frame = wicket_in_previous_frame
                wicket_in_previous_frame = wicket_in_this_frame

            # if wicket_in_this_frame == 9:
            #     if is_last_wicket(ocr_result):
            #         print('10 wicket')
            #         score.append((seconds, 10))
            #         wicket_in_this_frame = 10




        else:
            print(f"Error reading frame at {i / cap.get(cv2.CAP_PROP_FPS)} seconds")

    ocr_result_file.close()
    cap.release()
    return score


def extract_video_clip(input_path, output_path, start_time, end_time):
    """
    Extract a video clip from the input video.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the extracted clip.
        start_time (float): Start time of the clip.
        end_time (float): End time of the clip.
    """
    video_clip = VideoFileClip(input_path)
    clip = video_clip.subclip(start_time, end_time)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


def clipping_video(scores, input_video_path, output_folder):
    """
    Clip videos based on the wicket events.

    Args:
        scores (list): List of tuples containing time and wicket information.
        input_video_path (str): Path to the input video file.
        output_folder (str): Path to save the clipped videos.
    """
    wicket_one_frame_back = 0
    count = 0
    wicket = []

    for entry in scores:
        if (entry[1] - wicket_one_frame_back) == 1:
            wicket.append(entry[0])
            wicket_one_frame_back = entry[1]



    os.makedirs(output_folder, exist_ok=True)

    print(f'Total Wickets Found {len(wicket)}')

    for i in range(len(wicket)):
        output_video_path = os.path.join(output_folder, f"Wicket_video{count:d}.mp4")
        start_time_seconds = wicket[i] - 20
        end_time_seconds = wicket[i] - 1
        count += 1
        extract_video_clip(input_video_path, output_video_path,
                           start_time_seconds, end_time_seconds)

    return count


def process_video(video_path, extraction_interval, output_folder):
    """
    Process the input video.

    Args:
        video_path (str): Path to the input video file.
        extraction_interval (int): Interval for frame extraction.
        output_folder (str): Path to save the clipped videos.
    """
    print("Selecting a random frame...")
    # Select a random frame from the video
    random_frame = select_random_frame(video_path)
    print("Select an ROI to crop the frame.")
    # Allow the user to select an ROI for cropping
    cropped_frame, roi = crop_image(random_frame)

    # user give:
    score_position = input("Please enter the wicket position relative to hyphen(Left or Right) L/R: ")
    separator = input("Enter the separator: (Hyphen or Backslash) -//:")
    if(score_position== 'R' or score_position == 'L' or score_position== 'r' or score_position == 'l'):
        print("Applying the same crop to frames with specified interval in the video...")
        # Apply the same crop to frames in the video and extract wicket information
        score = apply_crop_to_video(video_path, roi, extraction_interval, score_position,separator)
        print("Cropping process completed.")

    else:
        print('Wrong input')
        return 0


    print("Clipping Videos")
    # Clip the videos based on the wicket events
    number_of_wicket = clipping_video(score, video_path, output_folder)
    print(number_of_wicket)
    print("Clipping Completed")


    # if last wicket 9






if __name__ == "__main__":
    input_video = "abhanivsshinepukur.mp4"
    output_folder = "wickets"
    interval = 100  # Interval for frame extraction

    start_time = time.time()
    process_video(input_video, interval, output_folder)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
