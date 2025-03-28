import os
import sys
import json
import logging
from neo4j import GraphDatabase

# Logging Configuration
LOG_DIRECTORY = "/tmp/netbird/log/"
LOG_FILENAME = "netbird-self-service2.log"

def setup_logging():
    """Sets up logging with specified log directory and file."""
    try:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(LOG_DIRECTORY, LOG_FILENAME),
            filemode='w'
        )
        logging.debug("Logging setup completed.")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

# Neo4j Connection Details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "azure_cli_kg"  # Change this to your desired database name

# Initialize Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def check_and_create_database():
    """Check if the Neo4j database exists, and create it if not."""
    try:
        with driver.session() as session:
            # Check if database exists
            result = session.run("SHOW DATABASES")
            existing_dbs = [record["name"] for record in result]

            if NEO4J_DATABASE in existing_dbs:
                logging.info(f"Database '{NEO4J_DATABASE}' already exists.")
            else:
                # Create the database
                session.run(f"CREATE DATABASE {NEO4J_DATABASE}")
                logging.info(f"Database '{NEO4J_DATABASE}' created successfully.")

    except Exception as e:
        logging.error(f"Error checking/creating database: {e}")
        sys.exit(1)

def load_json(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logging.info(f"Successfully loaded JSON file: {file_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
        return None

def create_knowledge_graph(tx, data):
    """Insert Command Group, Commands, Parameters, and Examples into Neo4j."""
    try:
        command_group = data['command_group']
        description = data['description']
        status = data['status']

        # Create Command Group Node
        tx.run("MERGE (cg:CommandGroup {name: $name, description: $desc, status: $status})", 
               name=command_group, desc=description, status=status)
        logging.info(f"Inserted CommandGroup: {command_group}")

        for cmd in data['commands']:
            cmd_name = cmd['name']
            summary = cmd['summary']
            syntax = cmd['syntax']

            # Create Command Node
            tx.run("""
                MERGE (c:Command {name: $cmd_name, summary: $summary, syntax: $syntax})
                MERGE (cg:CommandGroup {name: $cg_name})
                MERGE (cg)-[:HAS_COMMAND]->(c)
            """, cmd_name=cmd_name, summary=summary, syntax=syntax, cg_name=command_group)
            logging.info(f"Inserted Command: {cmd_name}")

            # Required Parameters
            for param in cmd.get('required_parameters', []):
                tx.run("""
                    MERGE (p:Parameter {name: $param, is_required: true})
                    MERGE (c:Command {name: $cmd_name})
                    MERGE (c)-[:HAS_REQUIRED_PARAMETER]->(p)
                """, param=param, cmd_name=cmd_name)
                logging.info(f"Inserted Required Parameter: {param} for {cmd_name}")

            # Optional Parameters
            for param in cmd.get('optional_parameters', []):
                tx.run("""
                    MERGE (p:Parameter {name: $param, is_required: false})
                    MERGE (c:Command {name: $cmd_name})
                    MERGE (c)-[:HAS_OPTIONAL_PARAMETER]->(p)
                """, param=param, cmd_name=cmd_name)
                logging.info(f"Inserted Optional Parameter: {param} for {cmd_name}")

            # Add Examples
            for example in cmd.get('examples', []):
                tx.run("""
                    MERGE (e:Example {command: $cmd_name, example_text: $example})
                    MERGE (c:Command {name: $cmd_name})
                    MERGE (c)-[:HAS_EXAMPLE]->(e)
                """, example=example, cmd_name=cmd_name)
                logging.info(f"Inserted Example for {cmd_name}: {example}")

    except Exception as e:
        logging.error(f"Error while creating knowledge graph: {e}")

def load_data_to_neo4j(json_files):
    """Load multiple JSON files into Neo4j."""
    with driver.session(database=NEO4J_DATABASE) as session:
        for file_path in json_files:
            logging.info(f"Processing file: {file_path}")
            data = load_json(file_path)
            if data:
                session.write_transaction(create_knowledge_graph, data)
            else:
                logging.warning(f"Skipping file {file_path} due to loading errors.")
    logging.info("Knowledge Graph Creation Completed.")

def main():
    setup_logging()

    # Check and create the database if it doesn't exist
    check_and_create_database()

    # Expecting at least one input parameter: JSON files
    if len(sys.argv) < 2:
        logging.error("Usage: python script.py <json_file1> <json_file2> ...")
        sys.exit(1)

    json_files = sys.argv[1:]  # All arguments after the script name are JSON file paths

    if not json_files:
        logging.error("No JSON files provided. Please specify at least one JSON file.")
        print("Error: No JSON files provided. Please specify at least one JSON file.")
        sys.exit(1)

    logging.info(f"Processing JSON files: {json_files}")

    # Load specified JSON files
    load_data_to_neo4j(json_files)

    # Close Neo4j Connection
    driver.close()
    logging.info("Neo4j connection closed.")

# ----------------------------------------------------------------------------
# SCRIPT EXECUTION
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
