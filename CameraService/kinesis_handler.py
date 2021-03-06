import boto3
import botocore.config
import json
import os
import random
import string
import config

aws_access_key = ""
aws_secret_access_key = ""
region = ""
data_stream_name = ""


def start_posting(data, images, debug=True):

    global aws_access_key
    global aws_secret_access_key
    global region
    global data_stream_name

    # data - dumped with json
    # images - path to pictures
    if os.environ['aws_access_key'] is not None:
        aws_access_key = os.environ['aws_access_key']

    if os.environ['aws_secret_access_key'] is not None:
        aws_secret_access_key = os.environ['aws_secret_access_key']

    if os.environ['region'] is not None:
        region = os.environ['region']

    if os.environ['data_stream_name'] is not None:
        data_stream_name = os.environ['data_stream_name']

    # initiate the kinesis stream connection
    kinesis_client = init_service_client("kinesis", aws_access_key, aws_secret_access_key, region)
    s3_client = init_service_client("s3", aws_access_key, aws_secret_access_key, region)

    # send the data to kinesis
    post_data_to_kinesis(kinesis_client, data_stream_name, data, debug)

    if images:
        # send the images path to post to kinesis
        post_images_to_s3(s3_client, images, debug)


def init_service_client(service, aws_access_key_id, aws_secret_access_key, region):

    config_client = botocore.config.Config()
    config_client.region_name = region
    config_client.connection_timeout = config.CONNECTION_TIMEOUT
    config_client.read_timeout = config.READ_TIMEOUT

    kinesis_client = boto3.client(service, config=config_client, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    return kinesis_client


def post_data_to_kinesis(client, data_stream_name, data, debug):

    print(json.dumps(json.loads(data), indent=4))

    post_to_kinesis(client, data_stream_name, data, debug)

    print("object has been posted!")


def post_images_to_s3(client, images_path, debug):

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
            Key = images_path + "/" + image
            outPutname = "stored_pics/{}".format(image)

            client.upload_file(Key, config.bucketName, outPutname)
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
        if int(response.get('ResponseMetadata').get('HTTPStatusCode')) == config.KINESIS_SUCCESS_RESPONSE:
            print(str(record) + " --> has been posted to kinesis")
        else:
            print(response)


def get_partition_key():

    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(config.NUMBER_OF_PARTITION_KEY_LETTERS))
