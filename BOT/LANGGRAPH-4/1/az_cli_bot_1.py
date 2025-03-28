import requests
import sys
import os
import logging

# ----------------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------------
LOG_DIRECTORY = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\BOT\LANGGRAPH-4\1\log'
LOG_FILENAME = "az_cli_bot_1.log"

def setup_logging():
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
            filemode='w'
        )
        # Optional: Log INFO to console too
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

setup_logging()

# ----------------------------------------------------------------------------
# GitHub Repository Details
# ----------------------------------------------------------------------------
GITHUB_OWNER = "vnyaryan"
GITHUB_REPO = "azure-docs"  # Corrected repo name

# Retrieve GitHub Token securely from environment variable
GITHUB_TOKEN = "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"


# GitHub Search API URL
GITHUB_SEARCH_API_URL = "https://api.github.com/search/code"

# ----------------------------------------------------------------------------
# Function to terminate script on error
# ----------------------------------------------------------------------------
def terminate(message):
    logging.error(message)
    sys.exit(1)

# ----------------------------------------------------------------------------
# Function to search keyword in GitHub repo
# ----------------------------------------------------------------------------
def search_keyword_in_repo(keyword):
    try:
        logging.info(f"Searching for keyword: {keyword}")
        
        query = f"{keyword}+repo:{GITHUB_OWNER}/{GITHUB_REPO}"
        headers = {
            "Accept": "application/vnd.github.text-match+json",
            "Authorization": f"token {GITHUB_TOKEN}"
        }
        params = {
            "q": query
        }

        response = requests.get(GITHUB_SEARCH_API_URL, headers=headers, params=params)

        if response.status_code != 200:
            error_msg = f"GitHub API Error {response.status_code}: {response.json().get('message')}"
            terminate(error_msg)

        results = response.json().get("items", [])
        if not results:
            logging.info(f"No results found for keyword '{keyword}'.")
            print(f"No results found for keyword '{keyword}'.")
            return

        print(f"\nSearch results for keyword: '{keyword}'\n{'-' * 50}")
        for item in results:
            file_path = item['path']
            html_url = item['html_url']
            print(f"File: {file_path}\nURL: {html_url}")

            log_msg = f"File: {file_path}, URL: {html_url}"
            logging.info(log_msg)

            text_matches = item.get('text_matches', [])
            for match in text_matches:
                fragment = match.get('fragment', '').strip()
                print(f">> Snippet: {fragment}")
                logging.debug(f"Snippet: {fragment}")
            print("-" * 50)

    except Exception as e:
        terminate(f"Exception occurred: {e}")

# ----------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        terminate("Usage: python github_keyword_search.py <keyword>")

    keyword_input = sys.argv[1]
    search_keyword_in_repo(keyword_input)
