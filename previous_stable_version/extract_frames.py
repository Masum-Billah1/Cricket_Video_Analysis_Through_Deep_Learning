import cv2
import os

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

def extract_frames(video_path, output_folder, interval=0.1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the video duration
    duration = get_video_duration(video_path)
    print(f"Video duration: {duration} seconds")

    # Calculate the frame interval in frames
    frame_interval = max(int(fps * interval), 1)

    # Initialize a counter for the number of frames saved
    frames_saved = 0

    # Iterate through frames at the specified interval
    for i in range(0, int(fps * duration), frame_interval):
        # Set the video capture to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read the frame
        ret, frame = cap.read()

        # If the frame is read successfully, save it
        if ret:
            seconds = i / fps

            output_path = os.path.join(output_folder, f"frame_{seconds:.1f}.jpg")

            
            # Save the frame
            cv2.imwrite(output_path, frame)
            print(f"Frame at {seconds:.1f} seconds saved to {output_path}")

            # Increment the counter
            frames_saved += 1
        else:
            print(f"Error reading frame at {i / fps} seconds")

    # Release the video capture object
    cap.release()

    # Return the number of frames saved
    return frames_saved

# if __name__ == "__main__":
#     # Specify the input video file, output folder, and other parameters
#     input_video = "input.mp4"
#     output_folder = "extracted_frames"
#     interval = 0.5 # Extract frame every 0.5 seconds

#     # Call the function to extract frames every 0.5 seconds
#     total_frames = extract_frames(input_video, output_folder, interval=interval)
#     print(f"Total number of frames saved: {total_frames}")
