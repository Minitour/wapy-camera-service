import numpy as np
import cv2
import config
import facial_landmarks
import os


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(config.object_pts, image_pts, config.cam_matrix, config.dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(config.reprojectsrc, rotation_vec, translation_vec, config.cam_matrix,
                                        config.dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def save_frame_as_picture(frame, object_id, frame_timestamp):

    # adding the timestamp and the x,y position we are attaching to the frame
    image_name = "{}{}_{}_{}_{}_{}.jpg".format(config.path_for_pictures, str(frame_timestamp), str(object_id), config.CAMERA_ID, config.STORE_ID, config.OWNER_UID)
    cv2.imwrite(image_name, frame)

    if config.DEBUG:
        print("saved photo with name: {}.jpg".format(image_name))


def clear_images_folder():
    for the_file in os.listdir(config.path_for_pictures):
        file_path = os.path.join(config.path_for_pictures, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


## -------------------------------------- handlers for analysis and values ------------------------------------ ##


def calc_normlized_dis(dis,angle):
    # temp_angle = 90 - np.deg2rad(angle)
    d = dis * np.sin(np.deg2rad(angle))
    if d == 0:
        d = 1
    return d


def normal_distances(diss, x_angle, y_angle):

    # getting the Y axe for extraction the normalized distance from camera
    first_normalization = []
    for dis in diss:
        d = calc_normlized_dis(dis, y_angle)
        first_normalization.append(d)

    # getting the normalized distance from camera (90 degrees angle from camera)
    final_normlization = []
    for normal in first_normalization:
        normal_dis = calc_normlized_dis(normal, x_angle)
        final_normlization.append(normal_dis)

    return final_normlization


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