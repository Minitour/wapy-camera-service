import requests
import json
import config
from requests.exceptions import RequestException
import helper_functions


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


# this function will get the mmo model from the json file
# saved on the disk
def get_json_model_from_mmo(path):

    if path == "":
        return stab_window_data()

    try:

        with open(path) as json_file:
            data = json.load(json_file)

            # means something went wrong and the json file is corrupted
            if data is None or (not data):
                return False, "no data found"

            # means there is no mmo data
            if data['mmo'] is None:
                return False, "no mmo data"

            # means there are no objects that were calibrated with window
            if data['mmo']['objects'] is None or (not data['mmo']['objects']):
                return False, "no objects were calibrated"

            return data['mmo']['objects']

    except IOError as error:
        if config.DEEP_DEBUG:
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

                # if we got here we
                return obj['object_id']

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
        object_left_point_distance = helper_functions.calculate_distance(positions['x'] - positions['r'], positions['y'])
        object_right_point_distance = helper_functions.calculate_distance(positions['x'] + positions['r'], positions['y'])
        object_up_point_distance = helper_functions.calculate_distance(positions['x'], positions['y'] - positions['r'])
        object_down_point_distance = helper_functions.calculate_distance(positions['x'], positions['y'] + positions['r'])

        if config.DEEP_DEBUG:
            print("object_left_point_distance: {}".format(object_left_point_distance))
            print("object_right_point_distance: {}".format(object_right_point_distance))
            print("object_up_point_distance: {}".format(object_up_point_distance))
            print("object_down_point_distance: {}".format(object_down_point_distance))

        new_mmo_data = {
            "object_id": object_id,                                     # string/int
            "object_left_point_distance": object_left_point_distance,   # float
            "object_right_point_distance": object_right_point_distance, # float
            "object_up_point_distance": object_up_point_distance,
            "object_down_point_distance": object_down_point_distance,
            "left_right": temp_left_right,                              # string "LEFT" or "RIGHT"
            "up_down": temp_up_down                                     # string "UP" or "DOWN"
        }

        if config.DEEP_DEBUG:
            print("new mmo data: {}".format(new_mmo_data))

        objects.append(new_mmo_data)

    # sorting the list for irrelevant objects
    filtered_by_left_right = [o for o in objects if o['left_right'] == left_right]
    filtered_by_up_down = [o for o in filtered_by_left_right if o['up_down'] == up_down]

    final_list = filtered_by_up_down

    return final_list