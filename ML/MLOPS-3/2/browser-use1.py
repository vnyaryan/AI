import os
import logging
import asyncio
from getpass import getpass  # Securely handle password input
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI

# Define constants for logging
log_directory = r'C:\Users\ARYAVIN\Documents\GitHub\AI\ai\ML\MLOPS-3\2'
log_filename = 'browser-use.log'

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

# Define OpenAI API key
OPENAI_API_KEY = "sk-proj-T1RLCxGczTeaypvM6WkuT3BlbkFJ6sf0MnB4MJr3PyCxQhOJ"

# ServiceNow URL
SERVICENOW_URL = "https://servicenow.i.mercedes-benz.com/now/nav/ui/classic/params/target/%24pa_dashboard.do%3F0aad0409472c3d109f8855d4316d43f2"
INCIDENT_NUMBER = "INC9481908"

# Get user credentials
username = "ARYAVIN"
password = "Rfvtgh0978"

# Initialize the browser
browser = Browser(config=BrowserConfig(headless=False))

# Define the task for the agent
task = f"""
1. Open {SERVICENOW_URL}.
2. Locate the 'User ID' field and enter '{username}'.
3. Click the 'Next' button to proceed to the password page.
4. Locate the 'Password' field and enter '{password}'.
5. Click the 'Log on' button to proceed.
6. Wait for the user to manually complete the two-factor authentication (2FA).
7. Display a message: "Please complete 2FA in your mobile app and press Enter when done."
8. Wait for the user confirmation before proceeding.
9. Locate the search bar on the ServiceNow dashboard.
10. Enter '{INCIDENT_NUMBER}' in the search bar.
11. Submit the search request.
12. Capture and display the search results in the terminal.
"""

# Initialize the agent
agent = Agent(
    task=task,
    llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY),
    browser=browser,
)

async def main():
    try:
        logging.info("Starting ServiceNow incident search with 2FA handling.")
        await agent.run()

        # Wait for user confirmation after 2FA
        input("Please complete the two-factor authentication in your mobile app, then press Enter to continue...")

        logging.info("User confirmed 2FA completion.")

        logging.info("Locating search bar and entering incident number.")
        print(f"Searching for incident: {INCIDENT_NUMBER}...")

        logging.info("Agent execution completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        input("Press Enter to close the browser...")
        await browser.close()
        logging.info("Browser closed.")

if __name__ == "__main__":
    asyncio.run(main())
