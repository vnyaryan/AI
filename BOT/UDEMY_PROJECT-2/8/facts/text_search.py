import os

# Constants
commands_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\7\facts\commands'
keyword = 'login'  # Replace with your specific keyword

# Function to search keyword in files
def search_keyword_in_files(directory, keyword):
    try:
        # List to store files containing the keyword
        files_with_keyword = []

        # Traverse through all files in the directory
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):  # Only process .txt files
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if keyword.lower() in content.lower():  # Case-insensitive search
                                files_with_keyword.append(file_path)
                                print(f"Keyword found in: {file_path}")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

        if not files_with_keyword:
            print(f"No files found with keyword '{keyword}'")
        else:
            print(f"Total files with keyword '{keyword}': {len(files_with_keyword)}")
            return files_with_keyword

    except Exception as e:
        print(f"Error while searching files: {e}")

# Main function
def main():
    search_keyword_in_files(commands_directory, keyword)

if __name__ == "__main__":
    main()
