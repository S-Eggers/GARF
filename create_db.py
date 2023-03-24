import sqlite3
import pandas as pd
from typing import Dict, Optional


def create_tables_for_dataset(connection: sqlite3.Connection, name: str, url: str):
    df = pd.read_csv(url, dtype=str)
    df.to_sql(f"{name}", connection, if_exists="replace", index=False)
    empty_df = df.iloc[0:0, :]
    empty_df.to_sql(f"{name}_copy", connection, if_exists="replace", index=False)
    empty_df.to_sql(f"{name}_copy1", connection, if_exists="replace", index=False)
    empty_df.to_sql(f"{name}_copy2", connection, if_exists="replace", index=False)

def create_database(datasets: Dict[str, str], database_url: Optional[str] = None):
    if database_url is None:
        connection = sqlite3.connect("database.db")
    else:
        connection = sqlite3.connect(database_url)

    for name, url in datasets.items():
        print(f"Creating tables for {name} from file at {url}...")
        create_tables_for_dataset(connection, name, url)

    connection.close()
    print("Database successfully created.")
    
    
if __name__ == '__main__':
    datasets = {"Hosp_rules": "./data/Hosp_clean.csv", "Food": "./data/Food_clean.csv"}
    create_database(datasets)
