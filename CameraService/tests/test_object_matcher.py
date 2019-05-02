############ this is a test for the functions that handle the object calculations and matching #############
# to use this test:
#          adjust the distance_to_object, x_angle, y_angle and array_of_lookers to values you want to test
#          and run in this section instead of main()
#
#          the mmo_data will be stab and can be changed in function stab_window_data()

import facial_landmarks
from facial_landmarks import get_axe_distance_to_object
from facial_landmarks import calculate_distance
from facial_landmarks import fit_model_object
from facial_landmarks import get_json_model_from_mmo

distance_to_object = 1


array_of_lookers = [
    [15,15],[15,16],[15,17],[15,18],[15,19],[15,20],[15,21],[15,22],[15,23],[15,24],[15,25],[15,26],[15,27],[15,28],[15,29],[15,30]
]

mmo_data_exists, mmo_data = get_json_model_from_mmo("","")

for o in array_of_lookers:
    print(o)
    x_angle = o[0]
    y_angle = o[1]

    # determine if the person wa looking to the left or right, up or down
    left_right = "RIGHT" if x_angle >= 0 else "LEFT"
    up_down = "UP" if y_angle >= 0 else "DOWN"

    # getting the distance from camera for each axe
    x_distance_camera_object = get_axe_distance_to_object(distance_to_object, x_angle)
    y_distance_camera_object = get_axe_distance_to_object(distance_to_object, y_angle)

    # getting the distance from camera to object
    camera_object_distance = calculate_distance(x_distance_camera_object, y_distance_camera_object)

    # check if the distance and direction fit any of the model object
    found_object = fit_model_object(mmo_data, camera_object_distance, left_right, up_down)

    if found_object != "":
        print("product found for angle: x_angle: {}, y_angle: {}, with id: {}".format(x_angle,y_angle,found_object))
    else:
        print("no product found for angles: x_angle:{}, y_angle: {}".format(x_angle, y_angle))
