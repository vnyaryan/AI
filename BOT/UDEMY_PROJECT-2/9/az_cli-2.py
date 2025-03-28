import os

def process_file(input_file):
    try:
        # Open the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()

        command_name = None
        description = None
        output_lines = []

        # Iterate through lines to find the keyword "Command" and extract the command name and description
        for i, line in enumerate(lines):
            if 'Command' in line:
                # Command keyword found, get the next line and extract the command name and description
                command_line = lines[i+1].strip()  # Get the next line after Command
                command_name = command_line.split(':')[0].strip()  # Extract text before ":"
                description = command_line.split(':')[1].strip()  # Extract text after ":"
                
                # Save command and description in the desired format
                output_lines.append(f"{command_name}")
                output_lines.append(f"{description}")

        # Check if a command name and description were found
        if command_name and description:
            # Define the output file name based on the input file
            output_file = input_file.replace('output1', 'output2')
            
            # Write the extracted command and description to a new file
            with open(output_file, 'w') as f:
                f.write("\n".join(output_lines))
            
            print(f"Command and description extracted and saved to {output_file}")
        else:
            print("No command found in the file.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Accept input file from user
    input_file = input("Enter the name of the input file: ")
    
    # Call the function to process the file
    process_file(input_file)
