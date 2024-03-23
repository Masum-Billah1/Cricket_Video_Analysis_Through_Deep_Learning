# Open the file in read mode
with open('scores.txt', 'r') as file:
    # Loop over each line in the file
    for line in file:
        # Print the current line
        text = line.strip()
        words = text.split('-')
        print(words)


        
        