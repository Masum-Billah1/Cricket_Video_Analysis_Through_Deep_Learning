import extract_frames
import crop_all_images
import extract_text_from_all_cropped_images
import finding_wicket_frame
def process_video(video_path, extraction_interval, crop_output_folder):
    
    # Step 1: Extract frames
    extracted_frames_folder = "extracted_frames"
    total_frames = extract_frames.extract_frames(video_path, extracted_frames_folder, extraction_interval)
    print(f"Total number of frames extracted: {total_frames}")
    
    # Step 2: Crop frames
    print("Select an image for cropping, then crop")
    crop_all_images.main(extracted_frames_folder, crop_output_folder)
    print("Cropping process completed.")

    # Step 3: Process cropped images for text detection
    print("Processing cropped images for text detection")
    extract_text_from_all_cropped_images.main(crop_output_folder)  # Assuming main() in text_detection processes the cropped images
    print("Text detection process completed.")

    #Step 4: Clip the video
    print("Clip the wicket videos")
    finding_wicket_frame.clipping_video()


    

if __name__ == "__main__":
    input_video = "input.mp4"
    crop_output_folder = "cropped_images"  # Output folder for cropped images
    interval = 1  # Interval for frame extraction

    process_video(input_video, interval, crop_output_folder)
