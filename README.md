# Camera-Service

The main service that runs on the WAPY-BOX.

This service has 4 sub-services: device-manager, data-handler, aws-handler and mmo-handler.

mmo-handler will access the mmo data that we modeled after activating the WAPY-BOX and adjust it to fit to our model.

device-manager will manage the camera - wapy-box connection, the service will start and init all the parameters
that the camera will need to detect the people.

data-handler will get the output data that the camera stream and transform it to values for the store.
the data-handler uses Dlib (python package) to get the facial landmarks of the person and RealSense SDK to get the distance
from that person, from there the handler manipulates the data according to the mmo data we got from the mmo-handler
to match the product the person was looking to the mmo model we constructed in the calibration service.

aws-handler will post the values we are getting from the data-handler to the AWS S3 bucket (will post to S3 only the images)
and to the Kinesis (only the data)

There are 2 types of data generated from this service, the name of the object detected and an image of the person looking at
specific object.

