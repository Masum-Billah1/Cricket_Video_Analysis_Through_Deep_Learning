import cv2
import os
import random

def select_random_image(folder_path):
    """
    Selects a random image from the specified folder and returns the image and its filename.
    """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    random_file = random.choice(files)
    return cv2.imread(os.path.join(folder_path, random_file)), random_file

def main_select_image(input_folder):
    while True:
        im, _ = select_random_image(input_folder)
        cv2.imshow("Image", im)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 13:  # Enter key
            return im
        else:
            print("Showing another random image...")

    
def crop_image(image):
    """
    Allows the user to crop the image and returns the cropped image and ROI.
    """
    roi = cv2.selectROI(image)
    cropped_image = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    cv2.destroyAllWindows()
    return cropped_image, roi


def apply_crop_to_folder(input_folder, output_folder, roi):
    """
    Applies the given ROI to crop all images in the input folder,
    and saves them in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cropped_image = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            cv2.imwrite(os.path.join(output_folder, file), cropped_image)
            print(f"{file_path} cropped")

def main(input_folder, output_folder):
    
    # input_folder = "extracted_frames"
    # output_folder = "cropped_images"
    
    # First, select the model image
    model_image = main_select_image(input_folder)
    cv2.imwrite("model_image.jpg", model_image)

    # Then, crop the model image
    cropped_image, roi = crop_image(model_image)

    # Finally, apply the same crop to all images in the input folder
    apply_crop_to_folder(input_folder, output_folder, roi)
    # print(f"All images have been cropped based on the model image and saved in '{output_folder}'.")

# if __name__ == "__main__":
#     main()
