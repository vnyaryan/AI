import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import logging

# Constants for logging and output
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\8\facts\logs'
log_filename = 'cli_processing.log'
output_directory = r'C:\Users\ARYAVIN\Documents\GitHub\BOT\UDEMY_PROJECT-2\8\facts'
output_filename = 'azcliurl.txt'

# Verbose flag
verbose = True

# Function to print messages if verbose is enabled
def vprint(message):
    if verbose:
        print(message)

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(log_directory, log_filename),
                            filemode='w')
        logging.debug("Logging setup completed.")
    except Exception as e:
        logging.error(f"Error setting up logging: {e}")

def extract_urls(webpage_url):
    vprint(f"Attempting to retrieve webpage: {webpage_url}")
    logging.info(f"Attempting to retrieve webpage: {webpage_url}")
    
    try:
        response = requests.get(webpage_url)
        vprint(f"Response status code: {response.status_code}")
        logging.info(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            logging.error(f"Failed to retrieve webpage: {response.status_code}")
            return []
        
        vprint("Webpage retrieved successfully.")
        logging.info("Webpage retrieved successfully.")

        # Save a sample of the HTML to verify content retrieval
        with open("sample_html.txt", "w", encoding="utf-8") as f:
            f.write(response.text[:1000])  # Save first 1000 characters for verification
        vprint("Sample HTML saved to sample_html.txt")
        
        # Parse the webpage content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        anchor_tags = soup.find_all('a')
        
        vprint(f"Found {len(anchor_tags)} anchor tags.")
        logging.info(f"Found {len(anchor_tags)} anchor tags.")

        urls = []
        for tag in anchor_tags:
            url = tag.get('href')
            if url:
                full_url = urljoin(webpage_url, url)
                urls.append(full_url)
        
        vprint(f"Extracted {len(urls)} URLs.")
        logging.info(f"Extracted {len(urls)} URLs.")
        return urls
    except Exception as e:
        vprint(f"Error retrieving or parsing webpage: {e}")
        logging.error(f"Error retrieving or parsing webpage: {e}")
        return []

def save_urls_to_file(urls, file_name):
    try:
        vprint(f"Creating output directory: {output_directory}")
        os.makedirs(output_directory, exist_ok=True)
        
        vprint(f"Saving {len(urls)} URLs to {file_name}")
        logging.info(f"Saving {len(urls)} URLs to {file_name}")
        
        with open(os.path.join(output_directory, file_name), 'w') as f:
            for url in urls:
                f.write(f"{url}\n")
        
        vprint(f"URLs saved to {file_name}.")
        logging.info(f"URLs saved to {file_name}.")
    except Exception as e:
        vprint(f"Error saving URLs to file: {e}")
        logging.error(f"Error saving URLs to file: {e}")

def main():
    vprint("Setting up logging...")
    setup_logging()  # Set up logging at the start of the program

    webpage_url = "https://learn.microsoft.com/en-us/cli/azure/account?view=azure-cli-latest"
    vprint(f"Starting URL extraction process for {webpage_url}")
    logging.info(f"Starting URL extraction process for {webpage_url}")

    urls = extract_urls(webpage_url)
    if urls:
        save_urls_to_file(urls, output_filename)
    else:
        vprint("No URLs found or failed to retrieve the webpage.")
        logging.warning("No URLs found or failed to retrieve the webpage.")

if __name__ == "__main__":
    main()
