# Open the file in read mode
substr = "-"
with open('scores.txt', 'r') as file:
    with open('scores2.txt', 'w') as output:
        # Loop over each line in the file
        for line in file:
            # Print the current line
            text = line.strip()
            if (text.__contains__(substr)):
                output.write(text+"\n")
                print(text)
