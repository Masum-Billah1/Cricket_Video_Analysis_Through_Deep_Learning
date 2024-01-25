import re

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

# Specify the file name
filename = "scores2.txt"

# Call the function to read, group, and sort the data
sorted_data = read_and_group_data(filename)

# print(sorted_data)

temp = 0
wicket = []
for entry in sorted_data:
    if temp!=entry[2]:
        wicket.append(entry[0])
        temp = entry[2]

print(wicket)
