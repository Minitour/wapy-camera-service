import boto3
import botocore.config
from base64 import b64encode
import json
import os
import random
import string

NUMBER_OF_PARTITION_KEY_LETTERS = 16
CONNECTION_TIMEOUT = 60
READ_TIMEOUT = 60

KINESIS_SUCCESS_RESPONSE = 200


def start_posting(aws_access_key_id, aws_secret_access_key, region, data_stream_name, images_stream_name, data, images, debug=False):

    # data - dumped with json
    # images - path to pictures

    # initiate the kinesis stream connection
    client = init_kinesis_client(aws_access_key_id, aws_secret_access_key, region)

    # send the data to kinesis
    post_data_to_kinesis(client, data_stream_name, data, debug)

    # send the images path to post to kinesis
    post_images_to_kinesis(client, images_stream_name, images, debug)


def init_kinesis_client(aws_access_key_id, aws_secret_access_key, region):

    config = botocore.config.Config()
    config.region_name = region
    config.connection_timeout = CONNECTION_TIMEOUT
    config.read_timeout = READ_TIMEOUT

    kinesis_client = boto3.client('kinesis', config=config, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    return kinesis_client


def post_data_to_kinesis(client, data_stream_name, data, debug):

    print("got " + str(len(json.loads(data))) + " tuple of points")

    index = 0
    for d in json.loads(data):

        post_to_kinesis(client, data_stream_name, json.dumps(d), debug)

        index += 1

    print("posted all " + str(index) + " data points\n")


def post_images_to_kinesis(client, images_stream_name, images_path, debug):

    general_error = None
    images = []
    counter_for_posting = 0

    # getting all images from path
    try:
        images = os.listdir(images_path)
    except Exception as error:
        general_error = error

    print("got " + str(len(images)) + " images from folder --> starting to post...")
    for image in images:

        try:
            with open(images_path + "/" + image, "rb") as image_file:

                # encode the image
                encoded_image = b64encode(image_file.read())

                # post the encoded image to kinesis
                post_to_kinesis(client, images_stream_name, encoded_image, debug)

                counter_for_posting += 1

        except Exception as error:
            general_error = error

    # if there is a problem we will print
    if general_error is not None:
        print(general_error)

    # print all the images we posted
    print("posted all " + str(counter_for_posting) + " images...\n" if counter_for_posting > 0 else "there are no images to post...\n")


def post_to_kinesis(client, stream_name, record, debug):

    response = client.put_record(StreamName=stream_name, Data=record, PartitionKey=get_partition_key())

    if debug:
        if int(response.get('ResponseMetadata').get('HTTPStatusCode')) == KINESIS_SUCCESS_RESPONSE:
            print(str(record) + " --> has been posted to kinesis")
        else:
            print(response)


def get_partition_key():

    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(NUMBER_OF_PARTITION_KEY_LETTERS))
