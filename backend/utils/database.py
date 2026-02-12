import sqlite3

class SQLiteDB:
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_name)

    def execute(self, query: str, *args):
        with self._get_connection() as con:
            cur = con.cursor()
            cur.execute(query, args)

    def fetchone(self, query: str, *args):
        with self._get_connection() as con:
            cur = con.cursor()
            cur.execute(query, args)
            return cur.fetchone()

    def fetchall(self, query: str, *args):
        with self._get_connection() as con:
            cur = con.cursor()
            cur.execute(query, args)
            return cur.fetchall()
        
db = SQLiteDB("database/data.db")

db.execute("""
CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    number BIGINT NOT NULL 
);
""")

#db.execute("""INSERT INTO emergency(name,number) VALUES("Aravind", 8111938885)""")
#TASKS = db.fetchall("SELECT task,time FROM reminders")
#print(TASKS)