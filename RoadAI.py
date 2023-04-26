#This code reads video files and splits it into frames snapshots and use image recognition model to extract details, store in csv file and postgreSql

#for batch processing model
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import psycopg2

# read the video file
cap = cv2.VideoCapture('video_file.mp4')

# create a TensorFlow model for image recognition
model = tf.keras.models.load_model('my_model.h5')

# connect to the PostgreSQL database
conn = psycopg2.connect(
    host="your_host",
    database="your_database",
    user="your_username",
    password="your_password"
)

# create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Frame'] + [f'Column{i}' for i in range(1, 21)])

while True:
    # get the next frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # split the frame into individual images
    frames = [frame[i:i+50] for i in range(0, frame.shape[0], 50)]
    
    # analyze each frame with the TensorFlow model
    for i, f in enumerate(frames):
        # pre-process the image
        img = cv2.resize(f, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        # make a prediction with the model
        prediction = model.predict(img)
        
        # extract the details from the prediction
        details = prediction[0][:20] # replace with the details you want to extract
        
        # create a dictionary of the results
        row_dict = {'Frame': i}
        for j, detail in enumerate(details):
            row_dict[f'Column{j+1}'] = detail
        
        # append the results to the DataFrame
        results_df = results_df.append(row_dict, ignore_index=True)

# save the results to a CSV file
results_df.to_csv('results.csv', index=False)

# append the results to the PostgreSQL database
with conn.cursor() as cur:
    for row in results_df.itertuples(index=False):
        cur.execute("""
            INSERT INTO results (frame, column1, column2, ..., column20)
            VALUES (%s, %s, %s, ..., %s);
        """, row)

conn.commit()
conn.close()

