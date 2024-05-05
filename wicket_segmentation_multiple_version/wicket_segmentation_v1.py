import cv2
import os
import random
import pytesseract
import numpy as np
import time
import re
from moviepy.video.io.VideoFileClip import VideoFileClip


# this method find the wicket from score
def find_digit_after_hyphen(text):
    # there will be least 1 hyphen because pattern is already matched
    hyphen_index = text.find('-')

    # if hyphen is not in the start
    if hyphen_index != 0:
        return text[hyphen_index + 1]

    next_hyphen_index = text.find('-', hyphen_index + 1)
    if next_hyphen_index != -1:
        return text[next_hyphen_index + 1]
    # No digit after hyphen
    return -1


def is_score(ocr_result):
    # Define the pattern for the score format
    pattern = r".*(\d{1,3})-(\d).*"

    # Use regular expressions to match the pattern
    match = re.match(pattern, ocr_result)

    # Return True if the pattern matches, False otherwise
    return match is not None


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


def get_video_duration(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) and frame count of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the duration in seconds
    duration = frame_count / fps

    # Release the video capture object
    cap.release()

    return duration


def select_random_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize variables for frame selection
    frame_selected = False
    frame = None

    while not frame_selected:
        # Select a random frame number
        random_frame_number = random.randint(0, total_frames - 1)

        # Set the frame position to the randomly selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

        # Read the selected frame
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow("Random Frame", frame)

        # Wait for user input
        key = cv2.waitKey(0) & 0xFF

        # Check if the user pressed 's' key
        if key == ord('s'):
            frame_selected = True
        else:
            print("Showing another random frame...")

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

    return frame


def crop_image(image):
    """
    Allows the user to crop the image and returns the cropped image and ROI.
    """
    roi = cv2.selectROI(image)
    cropped_image = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    cv2.destroyAllWindows()
    return cropped_image, roi


def apply_crop_to_video(video_path, output_folder, roi, extraction_interval):
    # score list that saves the score extracted from frames
    score = []
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize OCR result file
    ocr_result_file = open(os.path.join("ocr_results.txt"), "w")

    # Get the duration and fps from the whole video
    duration = int(get_video_duration(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the frame interval in frames
    frame_interval = max(int(fps * extraction_interval), 1)

    # Iterate through frames at the specified interval
    for i in range(0, duration * fps, frame_interval):
        # Set the video capture to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read the frame
        ret, frame = cap.read()

        # If the frame is read successfully, proceed with cropping and processing
        if ret:
            seconds = i / fps

            # Crop the frame using the ROI
            cropped_frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

            # Save the cropped frame
            output_path = os.path.join(output_folder, f"frame_{seconds:.1f}.jpg")
            cv2.imwrite(output_path, cropped_frame)

            # Perform OCR on cropped frame
            ocr_result = perform_ocr(cropped_frame)

            # Write OCR result to file
            if is_score(ocr_result):
                wicket_in_this_frame = find_digit_after_hyphen(ocr_result)
                ocr_result_file.write(f"Frame {seconds:.1f}: {wicket_in_this_frame}\n")
                # Check if wicket_in_this_frame is not equal to -1 and convert it to an integer
                if wicket_in_this_frame != -1:
                    wicket_in_this_frame = int(wicket_in_this_frame)
                score.append((seconds, wicket_in_this_frame))

        else:
            print(f"Error reading frame at {i / cap.get(cv2.CAP_PROP_FPS)} seconds")

    # Release the video capture object
    cap.release()

    # Close OCR result file
    ocr_result_file.close()
    return score


def perform_ocr(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform preprocessing steps as required
    gamma_transformed_image = PowerTransformation(gray_image)
    no_noise = noise_removal(gamma_transformed_image)
    dilated_image = thick_font(no_noise)
    eroded_image = thin_font(dilated_image)

    # Perform OCR
    custom_config = r'-l eng --oem 3 --psm 10 tessedit_char_whitelist=0123456789'
    ocr_result = pytesseract.image_to_string(eroded_image, config=custom_config)
    ocr_result = ocr_result.replace("\\", "").replace(" ", "").replace("o", "0").replace("O", "0").replace(",", "")
    return ocr_result.strip()


def extract_video_clip(input_path, output_path, start_time, end_time):
    # Load the video clip
    video_clip = VideoFileClip(input_path)

    # Extract the specified clip
    clip = video_clip.subclip(start_time, end_time)

    # Write the clip to a new file
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


def clipping_video(scores, input_video_path, output_folder):
    temp = 0
    count = 0
    wicket = []
    for entry in scores:
        if abs(temp - entry[1]) == 1:
            wicket.append(entry[0])
            temp = entry[1]

    os.makedirs(output_folder, exist_ok=True)

    print(f'len of wicket:{len(wicket)}')
    for i in range(len(wicket)):
        print(wicket[i])
    for i in range(len(wicket)):
        output_video_path = os.path.join(output_folder, f"Wicket_video{count:d}.mp4")
        start_time_seconds = wicket[i] - 25 # strat 25 second before score cahnge
        end_time_seconds = wicket[i] - 2 #end before 2 second scroe change
        count += 1
        extract_video_clip(input_video_path, output_video_path, start_time_seconds, end_time_seconds)


def process_video(video_path, extraction_interval, crop_output_folder, output_folder):
    # Step 1: Select a random frame
    print("Selecting a random frame...")
    random_frame = select_random_frame(video_path)

    # Step 2: Crop the random frame
    print("Select an ROI to crop the frame.")
    cropped_frame, roi = crop_image(random_frame)

    # Step 3: Apply the same crop to frames with specified interval in the video and get the wicket from each frame
    print("Applying the same crop to frames with specified interval in the video...")
    score = apply_crop_to_video(video_path, crop_output_folder, roi, extraction_interval)
    # Score is a list that contains pairs The first element of this pair is time_in_seconds of the video and the
    # second element is wicket at that moment shown on scoreboard
    print("Cropping process completed.")

    # Step 4 : Clipping Videos
    print("Clipping Videos")
    clipping_video(score, video_path, output_folder)
    print("Clipping Completed")


if __name__ == "__main__":
    input_video = "input2.mp4"
    crop_output_folder = "cropped_images"
    output_folder = "clipped_videos"
    interval = 1  # Interval for frame extraction

    # Record the start time
    start_time = time.time()

    # Call the process_video function
    process_video(input_video, interval, crop_output_folder, output_folder)

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the execution time
    print(f"Execution time: {execution_time} seconds")
