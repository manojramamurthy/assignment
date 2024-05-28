from fastapi import FastAPI, File, UploadFile
from typing import Annotated
import shutil
import os
from ultralytics import YOLO
import cv2
import pandas as pd
import sqlite3
import uuid

app = FastAPI()

def predict(image_path):
    img = cv2.imread(image_path)
    height = img.shape[0]
    width = img.shape[1]
    model = YOLO('weights2/best.pt')
    results = model([image_path])
    class_name = 'Coin'
    class_id = 0
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        
        xyxys = boxes.xyxy
        print(xyxys)
        
    return class_id, xyxys

def save_bboxes(data):
    # Connect to the SQLite database
    conn = sqlite3.connect('challenge1.db')

    file_name = data['file_name'][0]
    
    cur = conn.cursor()
    cur.execute(f"DELETE from uploaded_files where file_name='{file_name}'")
    
    # write data to sqlite3 datatable (challenge2)
    data.to_sql('uploaded_files', conn, if_exists='append', index=False)

    # Commit changes and close the connection
    conn.commit()
    conn.close()

def get_bboxes(imagename):
    # Connect to the SQLite database
    conn = sqlite3.connect('challenge1.db')
    # cur = conn.cursor()
    # result = cur.execute("SELECT xmin, ymin, xmax, ymax from uploaded_files")
    # result.fetchall()
    query = f"SELECT file_name, coin_id, xmin, ymin, xmax, ymax from uploaded_files where file_name = '{imagename.strip()}'"
    
    data = pd.read_sql_query(query, conn)
    
    conn.commit()
    conn.close()
    
    return data

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    if not file:
        return {"message": "No file sent"}
    else:
        return {"file_size:": len(file)}
    
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    if not (file.filename.endswith(".jpg") or file.filename.endswith(".png")):
        return {"Unknown file format, upload only .jpg or .png formats"}
    if not file:
        return {"message": "No upload file sent"}
    else:
        data = os.path.join("uploaded_images", file.filename)
        os.makedirs("uploaded_images", exist_ok=True)
        try:
            with open(data, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            file.file.close()
        
        df_final = pd.DataFrame(columns=['file_name', 'coin_id', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        class_id, bboxes = predict(data)
        coin_id = str(uuid.uuid4().hex[:8])
        for idx, bbox in enumerate(bboxes):
            df_final.loc[idx] = [str(file.filename), coin_id, str(class_id), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]
        
        if len(df_final) > 0:
            save_bboxes(df_final.copy())
        
        return {"filename": file.filename, "saved_path": data, "bboxes": df_final}

@app.post("/get_bounding_boxes/")
async def get_bounding_boxes(image_name: str):
    if not (image_name.endswith(".jpg") or image_name.endswith(".png")):
        return {"Unknown file format, upload only .jpg or .png formats"}
    else:
        data = get_bboxes(image_name)
        return {"data:": data}