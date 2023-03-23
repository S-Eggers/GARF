import sqlite3
import pandas as pd

df = pd.read_csv("./data/Hosp_clean.csv", dtype=str)
conn = sqlite3.connect("database.db")
df.to_sql("Hosp_rules_copy", conn, if_exists="replace", index=False)
df.to_sql("Hosp_rules", conn, if_exists="replace", index=False)
df.to_sql("Hosp_rules_copy1", conn, if_exists="replace", index=False)
df.to_sql("Hosp_rules_copy2", conn, if_exists="replace", index=False)

conn.close()