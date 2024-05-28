from fastapi import FastAPI
import pandas as pd
import sqlite3

app = FastAPI()

def get_bboxes(depth_min, depth_max):
    # Connect to the SQLite database
    conn = sqlite3.connect('challenge2.db')
    # cur = conn.cursor()
    # result = cur.execute("SELECT xmin, ymin, xmax, ymax from uploaded_files")
    # result.fetchall()
    query = f"SELECT * from challenge2 where depth >= '{depth_min.strip()}' and depth <= '{depth_max.strip()}'"
    
    data = pd.read_sql_query(query, conn)
    
    conn.commit()
    conn.close()
    
    return data

@app.post("/get_bounding_boxes/")
async def get_bounding_boxes(depth_min: str, depth_max):
    data = get_bboxes(depth_min, depth_max)
    return {"data:": data}