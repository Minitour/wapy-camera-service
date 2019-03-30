import cv2
import dlib
import numpy as np
from imutils import face_utils
import json
import math
import datetime
import os
import zipfile
import pyrealsense2 as rs
import re

DEBUG = True
POSSIBLE_CAMERAS = 1
DISTANCE = 1
NUMBER_OF_FRAMES_TO_SAVE_PICTURE = 100
NUMBER_OF_X_Y_TO_POST = 10
ALLOWED_X_Y_DISTANCE = 10

path_for_pictures = "./pictures_for_analysis/"
face_landmark_path = './shape_predictor_68_face_landmarks.dat'

##########################################################################################################
#                                                                                                        #
#                                                                                                        #
#                              CONSTANTS FOR POST PROCESSING FUNCTIONS                                   #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################


K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]



##########################################################################################################
#                                                                                                        #
#                                                                                                        #
#                                           DEVICES MANAGER                                              #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################


class Device:
    def __init__(self, pipeline, pipeline_profile):
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile


def enumerate_connected_devices(context):
    """
    Enumerate the connected Intel RealSense devices

    Parameters:
    -----------
    context 	 	  : rs.context()
                         The context created for using the realsense library

    Return:
    -----------
    connect_device : array
                       Array of enumerated devices which are connected to the PC

    """
    connect_device = []
    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera' and not re.search("(?<=d430).*", d.get_info(rs.camera_info.name).lower()):
            connect_device.append(d.get_info(rs.camera_info.serial_number))
    return connect_device


class DeviceManager:
    def __init__(self, context, pipeline_configuration):
        """
        Class to manage the Intel RealSense devices

        Parameters:
        -----------
        context 	: rs.context()
                                     The context created for using the realsense library
        pipeline_configuration 	: rs.config()
                                   The realsense library configuration to be used for the application

        """
        assert isinstance(context, type(rs.context()))
        assert isinstance(pipeline_configuration, type(rs.config()))
        self._context = context
        self._available_devices = enumerate_connected_devices(context)
        self._enabled_devices = {}
        self._config = pipeline_configuration
        self._frame_counter = 0


    def enable_device(self, device_serial):
        """
        Enable an Intel RealSense Device

        Parameters:
        -----------
        device_serial 	 : string
                             Serial number of the realsense device
        enable_ir_emitter : bool
                            Enable/Disable the IR-Emitter of the device

        """
        pipeline = rs.pipeline()

        # Enable the device
        self._config.enable_device(device_serial)
        pipeline_profile = pipeline.start(self._config)

        self._enabled_devices[device_serial] = (Device(pipeline, pipeline_profile))


    def enable_all_devices(self, enable_ir_emitter=False):
        """
        Enable all the Intel RealSense Devices which are connected to the PC

        """
        print(str(len(self._available_devices)) + " devices have been found")

        for serial in self._available_devices:
            self.enable_device(serial)


    def poll_frames(self, device_serial):
        """
        Poll for frames from the enabled Intel RealSense devices.
        If temporal post processing is enabled, the depth stream is averaged over a certain amount of frames

        Parameters:
        -----------

        """
        # getting the frames from the camera
        frames = self._enabled_devices[device_serial].pipeline.wait_for_frames()

        # extract the color and depth frames from the camera
        depth_frame = frames.get_depth_frame()
        color_frame = np.asanyarray(frames.get_color_frame().get_data())

        return color_frame, depth_frame


    def get_depth_shape(self):
        """ Retruns width and height of the depth stream for one arbitrary device

        Returns:
        -----------

        width: int
        height: int
        """
        width = -1
        height = -1
        for (serial, device) in self._enabled_devices.items():
            for stream in device.pipeline_profile.get_streams():
                if (rs.stream.depth == stream.stream_type()):
                    width = stream.as_video_stream_profile().width()
                    height = stream.as_video_stream_profile().height()
        return width, height


    def disable_streams(self):
        self._config.disable_all_streams()

    def get_serial_number(self, index):
        return self._available_devices[index]

    def get_available_camera(self):
        return len(self._available_devices)


def get_config_for_camera():

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    return config


def get_frames_from_all_cameras(device_manager, number_of_devices):

    frames = []

    for i in range(number_of_devices):
        serial = device_manager.get_serial_number(i)
        color_frame, depth_frame = device_manager.poll_frames(serial)
        frames.append({"device": str(serial), "color_frame": color_frame, "depth_frame": depth_frame})

    return frames


def main():
    # return
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--aws_access_key", required=True, help="the access key for aws")
    parser.add_argument("--aws_secret_access_key", required=True, help="the secret access key for aws")
    parser.add_argument("--region", required=True, help="the region for aws connection")
    parser.add_argument("--data_stream_name", required=True, help="the kinesis stream name for posting data")
    parser.add_argument("--images_stream_name", required=True, help="the kinesis stream name for posting images")

    args = parser.parse_args()

    # the config for the cameras
    config = get_config_for_camera()

    # init a device manager for camera/s
    device_manager = DeviceManager(rs.context(), config)

    # enable all devices we found
    device_manager.enable_all_devices()

    # adjust the number of available devices according to the number of cameras enabled successfully
    number_of_devices = device_manager.get_available_camera()

    # init the detector
    detector = dlib.get_frontal_face_detector()

    # init the predictor
    predictor = dlib.shape_predictor(face_landmark_path)

    # index when we will take the frame and get the emotions from the picture
    index = 1

    x_y_array = []

    while True:

        # getting all frames from cameras
        frames_per_camera = get_frames_from_all_cameras(device_manager, number_of_devices)

        width = 640
        height = 480
        ret = True if len(frames_per_camera) > 0 else False

        # we are still getting video from the camera
        if ret:
            color_frame = frames_per_camera[0]['color_frame']
            # getting the face rectangle from the frame
            face_rects = detector(color_frame, 0)

            # if we found some face/s in the frame
            if len(face_rects) > 0:

                # start to analyse the face/s in the frame
                for face in face_rects:
                    shape = predictor(color_frame, face)
                    shape = face_utils.shape_to_np(shape)

                    # estimate the head pose of the specific face
                    reprojectdst, euler_angle = get_head_pose(shape)

                    # draw points for the face
                    for (x, y) in shape:
                        cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)

                    # calculate where the observer is looking
                    x = width/2 - int((euler_angle[1, 0]/30)*(width/2)*(DISTANCE)) # the 30 parameter should be the angel up and down of the camera
                    y = height/2 + int((euler_angle[0, 0]/30)*(height/2)*(DISTANCE))

                    # after 1000 frames we are saving one photo
                    if index % NUMBER_OF_FRAMES_TO_SAVE_PICTURE == 0:
                        save_frame_as_picture(color_frame, x, y)

                    cv2.circle(color_frame, (int(x), int(y)), 3, (10, 20, 20), 2)

                    if DEBUG:
                        print("x: " + str(x) + ", y: " + str(y))

                    # appending the x,y to the list for posting to the messaging queue later
                    x_y_array = insert_x_y(x,y, x_y_array)

            cv2.imshow("WAPY", color_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                device_manager.disable_streams()
                break

        index += 1

        if index % NUMBER_OF_X_Y_TO_POST == 0:
            print("starting posting points and images to kinesis")

            # passing the array to the kinesis handler for posting
            import kinesis_handler
            kinesis_handler.start_posting(args.aws_access_key,              # aws access key
                                          args.aws_secret_access_key,       # secret access key
                                          args.region,                      # region
                                          args.data_stream_name,            # points stream name
                                          args.images_stream_name,          # images stream name
                                          json.dumps(x_y_array),            # array of points
                                          path_for_pictures)                # path to the pictures

            # init the array for the new posts
            x_y_array = []
            clear_images_folder()

            print("end of posting data and images --> new array of points started and images folder cleared")






##########################################################################################################
#                                                                                                        #
#                                                                                                        #
#                                   POST PROCESSING FUNCTIONS                                            #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################

def get_faces_from_detector(color_depth_frames, detector):
    faces = []

    for color_frame in color_depth_frames:

        color_frames = color_frame['color_frame']
        for frame in color_frames:
            face_rects = detector(frame, 0)

            faces.extend(face_rects)

    return faces

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def create_time_stamp():

    raw_timestamp = datetime.datetime.now()

    # example: raw_timestamp -> 2019-03-12 08:14:47.501562
    timestamp = str(raw_timestamp).split(".")[0].replace("-", "").replace(" ", "").replace(":", "")

    return timestamp


def save_frame_as_picture(frame, x, y):

    timestamp = create_time_stamp()

    # adding the timestamp and the x,y position we are attaching to the frame
    cv2.imwrite(path_for_pictures + timestamp + "_" + str(x) + "_" + str(y) + ".jpg", frame)

    if DEBUG:
        print("saved photo with timestamp:" + str(timestamp) + ".jpg")


def clear_images_folder():
    for the_file in os.listdir(path_for_pictures):
        file_path = os.path.join(path_for_pictures, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def insert_x_y(x,y, array_list):

    # indicator if we have already entered the point
    entered = False

    # checking if the current x,y is already in the json file
    # if so -> we will add to the number of views for that product
    for value in array_list:
        if value['x'] == x and value['y'] == y:

            # got the same x,y as in the json file and will add 1 to the value attribute
            entered = True
            value['value'] += 1
            break

    # if the current x,y is not in the json file -> will check if it's close enough to the product
    if not entered:
        array_list = check_close_pixel([x,y,1], array_list)

    # returning the new array list
    return array_list


def check_close_pixel(pxl, array_list):

    # checking if the current x,y is close to another pixel
    found = False
    for value in array_list:

        # we are checking if the x is in the allowed range of pixel to another pixel
        if pxl[0] + ALLOWED_X_Y_DISTANCE <= value['x'] or pxl[0] - ALLOWED_X_Y_DISTANCE >= value['x']:

            # we are checking if the y is in the allowed range of pixel to another pixel
            if pxl[1] + ALLOWED_X_Y_DISTANCE <= value['y'] or pxl[1] - ALLOWED_X_Y_DISTANCE >= value['y']:

                # add 1 to the pixel in the position we found
                value['value'] += pxl[2]
                found = True
                break

    # if we did not find any pixel close enough then
    # we will add to the array list the new x,y we got from the main function
    if not found:
        array_list.append({"x": pxl[0], "y": pxl[1], "value": pxl[2]})

    return array_list


# # function for adjust the list in order of value
# def finilize_to_send(list_to_order):
#
#     return list_to_order.sort(key=get_value)
#
#
# # getting the value of the current object
# def get_value(object):
#
#     if (not object) or (object is None):
#         return 0
#     return object['value']

if __name__ == '__main__':
    main()
