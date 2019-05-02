import cv2
import dlib
import numpy as np
from imutils import face_utils
import json
import requests
from requests.exceptions import RequestException
import datetime
from datetime import timedelta
import os
import subprocess
import pyrealsense2 as rs
import re

DEBUG = False
DEEP_DEBUG = False

# camera constants
POSSIBLE_CAMERAS = 1
UP_DOWN_ANGLE = 60
LEFT_RIGHT_ANGLE = 60
X_ANGLE = UP_DOWN_ANGLE / 2
Y_ANGLE = LEFT_RIGHT_ANGLE / 2

# post processing constants
DISTANCE = 1
NUMBER_OF_FRAMES_TO_SAVE_PICTURE = 10
NUMBER_OF_PRODUCTS_TO_POST = 100
TIME_DIFFERENCE_BEWTEEN_DETECTING = 4

ATTRIBUTE_NAME = "duration_so_far"

# constants for mmo data
headers = {"Content-Type": "application/json"}
SUCCESS = 200
NUMBER_OF_OBJECTS_IN_WINDOW = 0

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
        if d.get_info(rs.camera_info.name).lower() != 'platform camera' and not re.search("(?<=d430).*", d.get_info(
                rs.camera_info.name).lower()):
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
        self._profile_pipe = ""

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

        self._profile_pipe = pipeline_profile
        self._enabled_devices[device_serial] = (Device(pipeline, pipeline_profile))

    def enable_all_devices(self, enable_ir_emitter=False):
        """
        Enable all the Intel RealSense Devices which are connected to the PC

        """
        print("{} devices have been found".format(len(self._available_devices)))

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
        color_frame = color_frame.flip(color_frame, 0)

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

    def get_pipeline(self):
        return self._profile_pipe


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
    global DISTANCE
    global DEBUG
    global DEEP_DEBUG

    # return
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", help="debug level")
    parser.add_argument("--deep_debug", action="store_true", help="deeper debug level")

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

    # getting from the mmo the info we need for the window
    mmo_data_exists, mmo_data = get_json_model_from_mmo("", "")

    # init the product list with the json data for posting to kinesis
    product_list = []

    # set debug if stated
    DEBUG = args.debug if args.debug is not None else DEBUG
    print("debug mode: {}".format(DEBUG))

    DEEP_DEBUG = args.deep_debug if args.deep_debug is not None else DEEP_DEBUG
    print("deep debug mode, will print in console: {}".format(DEEP_DEBUG))

    while True:

        # getting all frames from cameras
        frames_per_camera = get_frames_from_all_cameras(device_manager, number_of_devices)

        width = 640
        height = 480
        ret = True if len(frames_per_camera) > 0 else False

        # we are still getting video from the camera
        if ret:
            color_frame = frames_per_camera[0]['color_frame']
            depth_frame = frames_per_camera[0]['depth_frame']
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

                    # init vars for measuring distance to object
                    dis_start = 0
                    dis_middle = 0
                    dis_end = 0

                    # getting the middle x,y of the shape detected
                    start_x, start_y = shape[0][0], shape[0][1]
                    middle_x, middle_y = shape[int(len(shape) / 2)][0], shape[int(len(shape) / 2)][1]
                    end_x, end_y = shape[-1][0], shape[-1][1]

                    if DEEP_DEBUG:
                        print("start x, y: {} , {}".format(start_x, start_y))
                        print("middle x, y: {} , {}".format(middle_x, middle_y))
                        print("end x, y: {} , {}".format(end_x, end_y))

                    # draw points for the face
                    for (x, y) in shape:

                        if DEEP_DEBUG:
                            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)

                    # getting the distance to the first x,y
                    if start_x is not None and start_y is not None:
                        dis_start = depth_frame.get_distance(start_x, start_y)

                    # getting the distance to the middle x,y
                    if middle_x is not None and middle_y is not None:
                        dis_middle = depth_frame.get_distance(middle_x, middle_y)

                    # getting the distance to the end x,y
                    if end_x is not None and end_y is not None:
                        dis_end = depth_frame.get_distance(end_x, end_y)

                    # getting the x, y angles from the euler angles
                    x_angle = euler_angle[1, 0]
                    y_angle = euler_angle[0, 0]

                    # getting the max distance -> avoiding 0
                    diss = [dis_start, dis_middle, dis_end]
                    #diss = [dis_end]

                    diss = normal_distances(diss, x_angle, y_angle)

                    # return value of too far objects is 0
                    # if no object detected because he is too far -> will put 1 as distance
                    distance_to_object = diss if diss is not None else DISTANCE

                    if DEEP_DEBUG:
                        print("distance to object: {}".format(distance_to_object))

                    # determine if the person wa looking to the left or right, up or down
                    left_right = "RIGHT" if x_angle >= 0 else "LEFT"
                    up_down = "UP" if y_angle <= 0 else "DOWN"

                    # getting the distance from camera for each axe
                    x_distances_camera_object = get_axe_distance_to_object(distance_to_object, x_angle)
                    y_distances_camera_object = get_axe_distance_to_object(distance_to_object, y_angle)
                    #print("\nx_distances_camera_object: {}".format(x_distances_camera_object))
                    #print("y_distances_camera_object: {}\n".format(y_distances_camera_object))
                    # getting the distance from camera to object
                    camera_object_distances = []
                    index_for_diss = 0
                    for x_dis in x_distances_camera_object:
                        camera_object_distance = calculate_distance(x_dis, y_distances_camera_object[index_for_diss])
                        camera_object_distances.append(camera_object_distance)
                        index_for_diss += 1

                    print(left_right)
                    print(up_down)
                    # check if the distance and direction fit any of the model object
                    found_object = fit_model_object(mmo_data, camera_object_distances, left_right, up_down)

                    if found_object != "":
                        # means we found an object that the observer was looking at
                        index += 1

                        # this function call will do one of 2 options:
                        # will update end timestamp if product detected -> will assign new datetime.now() -> extending time
                        # will init new timestamp for x,y with start_timestamp -> will reset if there is no faces detected
                        # until the next time some faces detected
                        product_list = insert_found_product(found_object, product_list)

                        #if DEEP_DEBUG:
                        print("found object with id: {}".format(found_object))

                    else:
                        #if DEEP_DEBUG:
                        print("no object found with distance from camera: {}".format(camera_object_distances))

            light_value = 0
            light_duration = ""
            dark_value = 0
            dark_duration = ""
            for p1 in product_list:
                if p1['product_id'] == "light":
                    light_value = p1['value']
                    light_duration = p1['duration_so_far']
                else:
                    dark_value = p1['value']
                    dark_duration = p1['duration_so_far']

            cv2.putText(color_frame, "LIGHT", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.putText(color_frame, "value: {}".format(light_value), (400, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.putText(color_frame, "duration: {}".format(light_duration), (400, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

            cv2.putText(color_frame, "DARK", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.putText(color_frame, "value: {}".format(dark_value), (80, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.putText(color_frame, "duration: {}".format(dark_duration), (80, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

            cv2.imshow("WAPY", color_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                device_manager.disable_streams()
                break

            # sum_up_values = int(light_value + dark_value) if int(light_value + dark_value) > 0 else 1
            # if sum_up_values % 100 == 0:
            #     timestamp = create_time_stamp()
            #     with open("./data_from_work/data_" + timestamp + ".json", "w") as json_file:
            #         json.dump(str(product_list), json_file)
            #     print("\nproduct list saved into json file with timestamp: {}\n".format(timestamp))

            if index % NUMBER_OF_PRODUCTS_TO_POST == 0:
                print("################################")
                for p in product_list:
                    print("{} : value={} , duration={}".format(p['product_id'], p['value'], p['duration_so_far']))
##########################################################################################################
#                                                                                                        #
#                                                                                                        #
#                                   POST PROCESSING FUNCTIONS                                            #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################


## -------------------------------------- face recognition helpers ------------------------------------ ##


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


## -------------------------------------- pictures proccessing helpers ------------------------------------ ##

def create_time_stamp():
    raw_timestamp = datetime.datetime.now()

    # example: raw_timestamp -> 2019-03-12 08:14:47.501562
    timestamp = str(raw_timestamp).split(".")[0].replace("-", "").replace(" ", "").replace(":", "")

    return timestamp


def save_frame_as_picture(frame, product):
    timestamp = create_time_stamp()

    # adding the timestamp and the x,y position we are attaching to the frame
    cv2.imwrite(path_for_pictures + timestamp + "_" + str(product) + ".jpg", frame)

    if DEBUG:
        print("saved photo with timestamp: {}.jpg".format(timestamp))


def clear_images_folder():
    for the_file in os.listdir(path_for_pictures):
        file_path = os.path.join(path_for_pictures, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


## -------------------------------------- handlers for analysis and values ------------------------------------ ##


def insert_found_product(product_id, product_list):

    '''
        product_list = [
            {
                "product_id": "",
                "value": "",
                "start_timestamp": "",
                "end_timestamp": "",
                "duration_so_far": "",
                "timers": []
            }
        ]
    '''

    # indicator if we have already entered the point
    entered = False

    # checking if the current product is already in the list
    # if so -> we will calculate the values for it: 'value', 'duration_so_far', timestamps...
    for product in product_list:

        if product['product_id'] == product_id:

            # save and remove the product from the list
            temp_product = product
            product_list.remove(product)

            # change the indicator that we found the object in the list
            entered = True

            # if we found the object in hand -> calculate the values and return the updated product
            updated_product = calculate_values(temp_product)
            product_list.append(updated_product)

            break

    # means this is a new product in the list -> append with init values
    if not entered:

        product_list.append({
            "product_id": product_id,
            "value": 1,
            "start_timestamp": datetime.datetime.now(),
            "end_timestamp": None,
            "duration_so_far": 0.0,
            "timers": []
        })

    return product_list


def calc_normlized_dis(dis,angle):
    temp_angle = 90 - np.deg2rad(angle)
    d = dis * np.sin(np.deg2rad(angle))
    if d == 0:
        d = 1
    return d


def normal_distances(diss, x_angle, y_angle):
    print(x_angle)
    print(y_angle)
    first_normalization = []
    for dis in diss:
        d = calc_normlized_dis(dis, y_angle)
        first_normalization.append(d)

    final_normlization = []
    for normal in first_normalization:
        normal_dis = calc_normlized_dis(normal, x_angle)
        final_normlization.append(normal_dis)

    # y_start_distance = get_axe_distance_to_object(diss, y_angle)
    # normalized_distances = get_axe_distance_to_object(y_start_distance, x_angle)

    return final_normlization

def calculate_values(value):
    global TIME_DIFFERENCE_BEWTEEN_DETECTING

    temp_value = value
    current_timestamp = datetime.datetime.now()

    if temp_value['end_timestamp'] is not None:
        if temp_value['end_timestamp'] + timedelta(seconds=TIME_DIFFERENCE_BEWTEEN_DETECTING) < current_timestamp:

            # end timestamp is not none but the session of looking is over -> need to calulate and start new session
            timer = {
                "start_timestamp": temp_value['start_timestamp'],
                "end_timestamp": temp_value['end_timestamp']
            }

            # calculate the time diff and convert into seconds to add to the duration_so_far
            delta = temp_value['end_timestamp'] - temp_value['start_timestamp']
            seconds_spent_on_product = delta.total_seconds()

            temp_value['value'] += 1

            # assign the values into object
            temp_value['duration_so_far'] += seconds_spent_on_product
            temp_value['timers'].append(timer)

            # starting new session -> need to put the current timestamp as start and None as end for init
            temp_value['start_timestamp'] = current_timestamp
            temp_value['end_timestamp'] = None

        else:
            # we are still in range of this session -> will put the current timestamp as end to
            # extend the time spent on this object
            temp_value['end_timestamp'] = current_timestamp

    elif temp_value['start_timestamp'] + timedelta(seconds=TIME_DIFFERENCE_BEWTEEN_DETECTING) < current_timestamp:
        # means we need to start new timer for this object
        # end_timestamp here is none
        temp_value['start_timestamp'] = current_timestamp

        temp_value['value'] += 1

    else:

        # means we need to update the end_timestamp to the current_timestamp to
        # extend the duration for the product when we calculate the values for json
        temp_value['end_timestamp'] = current_timestamp

    return temp_value


def get_axe_distance_to_object(distances_to_object, angle):
    x_distances = []

    for dis in distances_to_object:
        # calculate the angle in front of distance (90 - x_angle) -> distance_angle
        distance_angle = 90 - np.deg2rad(angle)

        # formula: distance / sin(distance_angle) = x_distance / sin(x_angle) ->
        # -> x_distance = distance * sin(x_angle) / sin(distance_angle)
        x_distance = dis * np.sin(np.deg2rad(angle)) / np.sin(np.deg2rad(distance_angle))
        x_distance = x_distance if x_distance != 0 else 1
        x_distances.append(x_distance)

    return x_distances


# this function will calculate the distance between the camera and the object
def calculate_distance(x_axe, y_axe):
    # sqrt(x_axe^2 + y_axe^2) -> from the imaginary triangle we created
    return np.sqrt(np.power(x_axe, 2) + np.power(y_axe, 2))


# function for adjust the list in order of value by attribute
def finilize_to_send(list_to_order):
    return list_to_order.sort(key=get_value)


# getting the value of the current object
def get_value(object):
    global ATTRIBUTE_NAME

    if (not object) or (object is None):
        return 0
    return object[ATTRIBUTE_NAME]


## -------------------------------------- handlers for handling model ------------------------------------ ##

def stab_window_data():
    return True, {
        "window": {
            "start": {
                "x": 0.2323,
                "y": 1.234,
                "z": 0.234
            },
            "end": {
                "x": -4.34,
                "y": 1.234,
                "z": -3.234
            }
        },
        "camera": {
            "euler": {
                "x": 0.2323,
                "y": 0.234,
                "z": 0.234
            }
        },
        "objects": [
            {
                "id": "dark",
                "position": {
                    "r": 0.15,
                    "x": 0.25,
                    "y": 0,
                    "z": 0
                }
            },
            {
                "id": "light",
                "position": {
                    "r": 0.15,
                    "x": -0.25,
                    "y": 0,
                    "z": 0
                }
            },
            {
                "id": "camera",
                "position": {
                    "r": 0.2,
                    "x": 0.01,
                    "y": 0.01,
                    "z": 0.01
                }
            }
        ]
    }


# will get with an api call
def get_json_model_from_mmo(auth, url):
    global SUCCESS
    global headers
    global NUMBER_OF_OBJECTS_IN_WINDOW

    if url == "":
        NUMBER_OF_OBJECTS_IN_WINDOW = 2
        return stab_window_data()

    try:
        response = requests.get(url, headers=headers, auth=auth)
        code = response.status_code

        if code == SUCCESS:
            body = json.loads(response.text)

            # setting the number of objects in window
            NUMBER_OF_OBJECTS_IN_WINDOW = len(body['objects'])

            return True, body

        else:

            return False, response.text

    except RequestException as error:
        if DEEP_DEBUG:
            print("error message: {}".format(error))
        return False, error.__str__()


def fit_model_object(mmo_data, camera_object_distance, left_right, up_down):
    destination_objects = get_camera_object_distance_mmo(mmo_data, left_right, up_down)

    '''
        destination_object = {
            "object_id": object_id,                                     # string/int
            "object_left_point_distance": object_left_point_distance,   # float
            "object_right_point_distance": object_right_point_distance, # float
            "object_up_point_distance": object_up_point_distance,
            "object_down_point_distance": object_down_point_distance,
            "left_right": temp_left_right,                              # string "LEFT" or "RIGHT"
            "up_down": temp_up_down                                     # string "UP" or "DOWN"
        }
    '''
    object_id = ""
    for obj in destination_objects:
        for camera_distance in camera_object_distance:

            # checking for range
            if obj['object_left_point_distance'] <= camera_distance <= obj['object_right_point_distance'] or obj['object_right_point_distance'] <= camera_distance <= obj['object_left_point_distance'] or obj['object_down_point_distance'] <= camera_distance <= obj['object_up_point_distance'] or obj['object_up_point_distance'] <= camera_distance <= obj['object_down_point_distance']:
                object_id = obj['object_id']
                break

    return object_id


def get_camera_object_distance_mmo(mmo_data, left_right, up_down):
    objects = []

    for m_d in mmo_data['objects']:

        object_id = m_d['id']

        # getting the values of the axes
        positions = m_d['position']

        # checking where the object if located by direction

        temp_left_right = "RIGHT" if positions['x'] >= 0.1 else "LEFT"
        temp_up_down = "DOWN" if positions['y'] <= 0 else "UP"

        # we will calculate the distance from the end points of the object and get the max and min distances
        object_left_point_distance = calculate_distance(positions['x'] - positions['r'], positions['y'])
        object_right_point_distance = calculate_distance(positions['x'] + positions['r'], positions['y'])
        object_up_point_distance = calculate_distance(positions['x'], positions['y'] - positions['r'])
        object_down_point_distance = calculate_distance(positions['x'], positions['y'] + positions['r'])

        if DEEP_DEBUG:
            print("object_left_point_distance: {}".format(object_left_point_distance))
            print("object_right_point_distance: {}".format(object_right_point_distance))
            print("object_up_point_distance: {}".format(object_up_point_distance))
            print("object_down_point_distance: {}".format(object_down_point_distance))

        new_mmo_data = {
            "object_id": object_id,  # string/int
            "object_left_point_distance": object_left_point_distance,  # float
            "object_right_point_distance": object_right_point_distance,  # float
            "object_up_point_distance": object_up_point_distance,
            "object_down_point_distance": object_down_point_distance,
            "left_right": temp_left_right,  # string "LEFT" or "RIGHT"
            "up_down": temp_up_down  # string "UP" or "DOWN"
        }

        #if DEEP_DEBUG:
        print("new mmo data: {}".format(new_mmo_data))

        objects.append(new_mmo_data)

    # sorting the list for irrelevant objects
    filtered_by_left_right = [o for o in objects if o['left_right'] == left_right]
    filtered_by_up_down = [o for o in filtered_by_left_right if o['up_down'] == up_down]

    final_list = filtered_by_up_down

    return final_list


if __name__ == '__main__':
    main()
