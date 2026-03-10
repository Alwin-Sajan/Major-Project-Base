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
            return cur.lastrowid

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
CREATE TABLE IF NOT EXISTS student (
    sid INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    institution TEXT NOT NULL,
    password TEXT NOT NULL
);
""")

db.execute("""
CREATE TABLE IF NOT EXISTS admin (
    aid INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL
);
""")
 
db.execute("""
CREATE TABLE IF NOT EXISTS leaderboard (
    sid INTEGER PRIMARY KEY ,
    score INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (sid) REFERENCES student(sid) ON DELETE CASCADE
);
""")



