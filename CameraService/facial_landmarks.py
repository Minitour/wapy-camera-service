import cv2
import dlib
import numpy as np
from imutils import face_utils
import json
import math
import datetime
import os
import zipfile

DEBUG = True
POSSIBLE_CAMERAS = 1
DISTANCE = 1
NUMBER_OF_FRAMES_TO_SAVE_PICTURE = 200
NUMBER_OF_X_Y_TO_POST = 500
ALLOWED_X_Y_DISTANCE = 10

path_for_pictures = "./pictures_for_analysis/"
face_landmark_path = './shape_predictor_68_face_landmarks.dat'


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


def get_available_cameras():

    cameras = []

    for index_for_camera in range(0, POSSIBLE_CAMERAS):

        try:
            # trying to connect to camera with specific index
            camera = cv2.VideoCapture(index_for_camera)

            # append to available cameras
            if camera:
                print("got camera: " + str(camera))
                cameras.append(camera)

        except Exception as error:
            if DEBUG:
                print(error)

    # returning the available cameras
    return cameras


# default cameras:
    # index == 0 -> the computer camera
    # index == 1 -> external camera
    # index > 1 -> other cameras -> we will not implement more than one camera BUT SUPPORTED
def get_default_camera(cameras, index_for_camera=0, external=False):

    if len(cameras) == 0:
        return None

    if len(cameras) == 1 and not external:
        return cameras[index_for_camera]

    if len(cameras) == 1 and external:
        return None

    if len(cameras) > 1 and not external:
        return cameras[index_for_camera]

    if external:
        return cameras[index_for_camera]


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



def main():
    # return

    external = False

    # getting all available cameras that are connected to the compute stick
    cameras = get_available_cameras()

    # getting the camera we want to use -> default => the computer camera
    # to get the external camera you can add True parameter to this function and also select the number of the camera
    # if you do not specific any index/external parameter -> will get the default one (computer)
    # example: cap = get_default_camera(cameras, 1, True)
    cap = get_default_camera(cameras, external)

    # if the external param is True -> cap might be equal to [] if the function cant find a connected camera
    if (cap is None) or (not cap.isOpened()):
        print(cap)
        print("Unable to find camera.")
        return ["NO_CAMERAS"]

    # init the detector
    detector = dlib.get_frontal_face_detector()

    # init the predictor
    predictor = dlib.shape_predictor(face_landmark_path)

    # index when we will take the frame and get the emotions from the picture
    index = 1

    x_y_array = []

    while cap.isOpened():

        ret, frame = cap.read()

        # getting the width and height from the video
        width = cap.get(3)
        height = cap.get(4)

        if DEBUG:
            print(width)
            print(height)

        # we are still getting video from the camera
        if ret:

            # getting the face rectangle from the frame
            face_rects = detector(frame, 0)

            # if we found some face/s in the frame
            if len(face_rects) > 0:

                # start to analyse the face/s in the frame
                for face in face_rects:
                    shape = predictor(frame, face)
                    shape = face_utils.shape_to_np(shape)

                    # estimate the head pose of the specific face
                    reprojectdst, euler_angle = get_head_pose(shape)
                    #print(reprojectdst)

                    # draw points for the face
                    for (x, y) in shape:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                    # calculate where the observer is looking
                    x = width/2 - int((euler_angle[1, 0]/30)*(width/2)*(DISTANCE)) # the 30 parameter should be the angel up and down of the camera
                    y = height/2 + int((euler_angle[0, 0]/30)*(height/2)*(DISTANCE))

                    # after 1000 frames we are saving one photo
                    if index % NUMBER_OF_FRAMES_TO_SAVE_PICTURE == 0:
                        save_frame_as_picture(frame, x, y)

                    cv2.circle(frame, (int(x), int(y)), 3, (10, 20, 20), 2)

                    if DEBUG:
                        print("x: " + str(x) + ", y: " + str(y))

                    # appending the x,y to the list for posting to the messaging queue later
                    x_y_array.append((x,y))

            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        index += 1

        if index % NUMBER_OF_X_Y_TO_POST == 0:
            print("posting the x,y array to the messaging queue...")



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
        array_list.append({"x": pxl[0], "y": pxl[1], "value": pxl[2], "radius": 40})

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