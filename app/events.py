import sqlite3
from pathlib import Path
class EventStore:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True);self.path=path
        with self.connect() as c:c.execute("CREATE TABLE IF NOT EXISTS events (user INTEGER, item INTEGER, rating REAL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(user,item))")
    def connect(self):
        c=sqlite3.connect(self.path,timeout=10);c.execute("PRAGMA journal_mode=WAL");return c
    def record(self,user,item,rating):
        with self.connect() as c:c.execute("INSERT INTO events(user,item,rating) VALUES(?,?,?) ON CONFLICT(user,item) DO UPDATE SET rating=excluded.rating, updated_at=CURRENT_TIMESTAMP",(user,item,rating))
    def history(self,user):
        with self.connect() as c:return c.execute("SELECT item,rating FROM events WHERE user=? ORDER BY item",(user,)).fetchall()
