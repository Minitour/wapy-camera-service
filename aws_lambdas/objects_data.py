import sys
import pymysql
import os
import base64
import json


def get_connection():
    conn = ""
    try:
        conn = pymysql.connect(os.environ['RDS_HOST'], user=os.environ['DB_USERNAME'], passwd=os.environ['DB_PASSWORD'],
                               db=os.environ['DB_NAME'], connect_timeout=5)
    except Exception as e:
        print(e)
        print("ERROR: Unexpected error: Could not connect to MySQL instance.")
        sys.exit()

    print("SUCCESS: Connection to RDS MySQL instance succeeded")
    return conn


def handler(event, context):
    print(event)
    records = event['Records']

    for record in records:
        # decode the data from base64 to json object
        decoded_object = base64.b64decode(record['kinesis']['data'])
        json_object = json.loads(decoded_object)

        # get the fields
        store_id = json_object['store_id']
        camera_id = json_object['camera_id']
        object_id = json_object['object_id']
        timestamp = json_object['timestamp']

        values = "{},{},{},{}".format(store_id, camera_id, object_id, timestamp)

        fields = "store_id, camera_id, object_id, timestamp"

        query = "INSERT INTO {} ({}) VALUES ({})".format(os.environ['DB_TABLE'], fields, values)
        print(query)
        # getting connection to my sql database
        conn = get_connection()
        if conn == "":
            sys.exit(1)

        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
            print("committed")
