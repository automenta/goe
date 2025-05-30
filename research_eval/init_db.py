import sqlite3
from sqlite3 import Error
import os
import logging

# Setup logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the database file path (in the project root for simplicity)
DB_FILE = "experiment_results.db"

def create_connection(db_file):
    """ Create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"Connected to SQLite database: {db_file} (SQLite version {sqlite3.sqlite_version})")
    except Error as e:
        logging.error(f"Error connecting to database {db_file}: {e}")
    return conn

def create_table(conn, create_table_sql):
    """ Create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        table_name = create_table_sql.split("TABLE IF NOT EXISTS ")[1].split(" (")[0]
        logging.info(f"Table '{table_name}' created successfully (or already exists).")
    except Error as e:
        logging.error(f"Error creating table: {e}")

def initialize_database(db_file=DB_FILE):
    """Create and initialize the SQLite database and its tables."""
    
    sql_create_experiments_table = """ CREATE TABLE IF NOT EXISTS experiments (
                                        experiment_id TEXT PRIMARY KEY,
                                        run_type TEXT NOT NULL,
                                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                        experiment_name TEXT,
                                        model_type TEXT NOT NULL,
                                        dataset_name TEXT NOT NULL,
                                        dataset_type TEXT NOT NULL,
                                        epochs INTEGER NOT NULL,
                                        batch_size INTEGER NOT NULL,
                                        full_config_json TEXT,
                                        env_details_json TEXT
                                    ); """

    sql_create_metrics_table = """CREATE TABLE IF NOT EXISTS metrics (
                                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    experiment_id TEXT NOT NULL,
                                    metric_name TEXT NOT NULL,
                                    metric_value REAL, 
                                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE
                                );"""

    conn = create_connection(db_file)

    if conn is not None:
        create_table(conn, sql_create_experiments_table)
        create_table(conn, sql_create_metrics_table)
        
        # Add indexes for frequently queried columns
        try:
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_run_type ON experiments (run_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_model_type ON experiments (model_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_dataset_name ON experiments (dataset_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric_experiment_id ON metrics (experiment_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics (metric_name);")
            logging.info("Indexes created successfully (or already exist).")
        except Error as e:
            logging.error(f"Error creating indexes: {e}")
        
        conn.close()
        logging.info(f"Database {db_file} initialized and tables created/verified.")
    else:
        logging.error("Error! Cannot create the database connection.")

if __name__ == '__main__':
    initialize_database()
    logging.info(f"Database '{DB_FILE}' is ready for use.")
