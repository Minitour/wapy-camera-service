import boto3
import json
import os
import sys
import pymysql


def db_connect():
    conn = ""
    try:
        conn = pymysql.connect(os.environ['RDS_HOST'], user=os.environ['DB_USERNAME'], passwd=os.environ['DB_PASSWORD'],
                               db=os.environ['DB_NAME'], connect_timeout=5)
        print("SUCCESS: Connection to RDS MySQL instance succeeded")
    except Exception as e:
        print(e)
        print("ERROR: Unexpected error: Could not connect to MySQL instance.")
        sys.exit()
    return conn


def detect_face_labels(fileName):
    client = boto3.client('rekognition')

    response = client.detect_faces(Attributes=["ALL", "DEFAULT"],
                                   Image={'S3Object': {'Bucket': os.environ['BUCKET'], 'Name': fileName}})

    labels = {}

    # age range in form of low and high
    print(response['FaceDetails'][0]['AgeRange'])
    labels.update({"age_low": response['FaceDetails'][0]['AgeRange']['Low']})
    labels.update({"age_high": response['FaceDetails'][0]['AgeRange']['High']})

    # gender in form of M-male, F-female
    print(response['FaceDetails'][0]['Gender'])
    labels.update({"gender": str(response['FaceDetails'][0]['Gender']['Value']).lower()[0]})

    # smile - if the person if smiling
    labels.update({"smile": response['FaceDetails'][0]['Smile']['Value']})
    print(response['FaceDetails'][0]['Smile'])

    # other emotions
    print(response['FaceDetails'][0]['Emotions'])
    emotions = response['FaceDetails'][0]['Emotions']
    for v in emotions:
        labels.update({str(v['Type']).lower(): v['Confidence']})

    return labels


def process_image_name(name):
    # frame_timestamp, object_id, CAMERA_ID, STORE_ID
    raw = str(name).split(".")[0]
    values = str(raw).split("_")
    timestamp = values[0]
    object_id = values[1]
    camera_id = values[2]
    store_id = values[3]
    print("{}, {}, {}, {}".format(timestamp, object_id, camera_id, store_id))
    return timestamp, store_id, camera_id, object_id


def process(event, context):
    print(event)

    # getting the file name from the event in s3
    record = event['Records'][0]['s3']['object']['key']
    record_for_processing = str(record).split("/")[1]

    # process the image name for params
    timestamp, store_id, camera_id, object_id = process_image_name(record_for_processing)

    print(record)

    try:

        # send the image to rekognition to get emotions and other data
        anots = detect_face_labels(record)
        print(anots)
        age_low = anots['age_low']
        age_high = anots['age_high']
        smile = anots['smile']
        gender = anots['gender']
        calm = anots['calm']
        happy = anots['happy']
        surprised = anots['surprised']
        confused = anots['confused']
        disgusted = anots['disgusted']
        angry = anots['angry']
        sad = anots['sad']

        # construct the values for query
        values = "{},{},{},{},{},{},{},{},{},{},{},{},{},'{}',{}".format(store_id, camera_id, object_id, timestamp,
                                                                         age_low, age_high, smile, calm, happy,
                                                                         confused, disgusted, angry, sad,
                                                                         gender, surprised)

        # construct the fields for query
        fields = "store_id, camera_id, object_id, timestamp, age_low, age_high, smile, calm, happy, confused, disgusted, angry, sad, gender, surprised"

        query = "INSERT INTO {} ({}) VALUES ({})".format(os.environ['DB_TABLE'], fields, values)
        print(query)

        # getting connection to my sql db
        conn = db_connect()
        if conn == "":
            sys.exit(1)

        # posting the data to the my sql table
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
            print("committed")
            cur.execute("select * from {}".format(os.environ['DB_TABLE']))
            for row in cur:
                print(row)

    except Exception as ex:
        print('EXCEPTION: {}'.format(ex))
