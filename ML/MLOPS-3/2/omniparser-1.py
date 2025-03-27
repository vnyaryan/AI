import os
import logging
import asyncio
import time
from omniparser import OmniParser
from PIL import ImageGrab  # For taking screenshots
import pyautogui  # For interacting with detected elements

# Define constants for logging
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\2'
log_filename = 'omniparser-1.log'

# Function to set up logging
def setup_logging():
    try:
        os.makedirs(log_directory, exist_ok=True)
        with open(os.path.join(log_directory, log_filename), 'w') as log_file:
            log_file.write("# Log file for ServiceNow 2FA login automation\n")
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(log_directory, log_filename),
            filemode='a'
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

# Call logging setup function
setup_logging()

# ServiceNow details
SERVICENOW_URL = "https://servicenow.i.mercedes-benz.com/now/nav/ui/classic/params/target/%24pa_dashboard.do%3F0aad0409472c3d109f8855d4316d43f2"
INCIDENT_NUMBER = "INC9481908"

# User credentials
username = "ARYAVIN"
password = "Rfvtgh0978"

# Initialize OmniParser
parser = OmniParser()

# Function to take a screenshot
def take_screenshot(filename="screenshot.png"):
    img = ImageGrab.grab()
    img.save(filename)
    return filename

# Function to detect elements using OmniParser
def detect_and_interact(element_name, input_text=None):
    screenshot = take_screenshot()
    results = parser.parse(screenshot)

    for result in results:
        if element_name.lower() in result['description'].lower():
            x, y = result['position']
            pyautogui.click(x, y)
            if input_text:
                pyautogui.write(input_text)
            return True

    logging.error(f"Could not find {element_name} on screen.")
    return False

# Main workflow
async def main():
    try:
        logging.info("Starting ServiceNow login with OmniParser.")

        # Open ServiceNow in browser (Manually done by user)
        os.system(f"start {SERVICENOW_URL}")  # Opens in default browser
        time.sleep(5)  # Wait for page to load

        # Enter User ID
        logging.info("Entering User ID...")
        if detect_and_interact("User ID", username):
            logging.info("User ID entered successfully.")

        # Click Next
        if detect_and_interact("Next"):
            logging.info("Clicked Next button.")
        time.sleep(3)

        # Enter Password
        logging.info("Entering Password...")
        if detect_and_interact("Password", password):
            logging.info("Password entered successfully.")

        # Click Log on
        if detect_and_interact("Log on"):
            logging.info("Clicked Log on button.")
        time.sleep(3)

        # Wait for user to complete 2FA manually
        input("Please complete 2FA on your mobile device, then press Enter to continue...")

        logging.info("User confirmed 2FA completion.")
        time.sleep(3)

        # Locate Search Bar & Enter Incident Number
        logging.info(f"Searching for Incident {INCIDENT_NUMBER}...")
        if detect_and_interact("Search", INCIDENT_NUMBER):
            logging.info("Incident number entered successfully.")

        logging.info("Search completed.")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        logging.info("Script execution completed.")

if __name__ == "__main__":
    asyncio.run(main())
