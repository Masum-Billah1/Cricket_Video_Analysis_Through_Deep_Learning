import re
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_video_clip(input_path, output_path, start_time, end_time):
    # Load the video clip
    video_clip = VideoFileClip(input_path)

    # Extract the specified clip
    clip = video_clip.subclip(start_time, end_time)

    # Write the clip to a new file
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


def extract_numbers_from_string(input_string):
    # Define the pattern for extracting numbers
    pattern = r"Score from frame_(\d+\.\d+)\.jpg: (\d+)-(\d+)"

    # Use re.match to find the pattern in the input string
    match = re.match(pattern, input_string)

    # Check if a match is found
    if match:
        # Extract the numbers from the matched groups
        frame_number = float(match.group(1))
        score1 = int(match.group(2))
        score2 = int(match.group(3))

        return frame_number, score1, score2

def read_and_group_data(filename):
    # Initialize an empty list to store the data as a 3D array
    data_3d_array = []

    # Read data from the file
    with open(filename, 'r') as file:
        for line in file:
            # Extract numbers from each line
            result = extract_numbers_from_string(line)

            # If numbers are extracted, append them to the data_3d_array
            if result:
                data_3d_array.append(result)

    # Sort the data based on the "Frame Numbers"
    data_3d_array.sort(key=lambda x: x[0])

    return data_3d_array

def clipping_video():
    # Specify the file name
    filename = "scores.txt"

    # Call the function to read, group, and sort the data
    sorted_data = read_and_group_data(filename)
    # print(sorted_data)

    temp = 0
    count = 0
    wicket = []
    for entry in sorted_data:
        if abs(temp-entry[2])== 1:
            wicket.append(entry[0])
            temp = entry[2]

    input_video_path = "input.mp4"

    # Create a directory to store the clipped videos if it doesn't exist
    output_folder = "clipped_videos"
    os.makedirs(output_folder, exist_ok=True)

    print(f'len of wicket:{len(wicket)}')
    for i in range(len(wicket)):
        print(wicket[i])
    for i in range(len(wicket)):
        output_video_path = os.path.join(output_folder, f"Wicket_video{count:d}.mp4")
        start_time_seconds = wicket[i] - 10
        end_time_seconds = wicket[i]
        count += 1
        extract_video_clip(input_video_path, output_video_path, start_time_seconds, end_time_seconds)
