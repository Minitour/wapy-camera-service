import cv2
import dlib
from imutils import face_utils
import json
import os
import kinesis_handler
import time
import config
import device_manager_class
import helper_functions
import mmo_handler
import datetime


def main():

    if os.environ['MMO_PATH'] is not None:
        config.change_mmo_path(os.environ['MMO_PATH'])

    # the config for the cameras
    config_pipeline = device_manager_class.get_config_for_camera()

    # init a device manager for camera/s
    device_manager = device_manager_class.DeviceManager(config_pipeline)

    # enable all devices we found
    device_manager.enable_all_devices()

    # adjust the number of available devices according to the number of cameras enabled successfully
    number_of_devices = device_manager.get_available_camera()

    if number_of_devices < 1:
        print("No cameras connected.")
        return 1

    # init the detector
    detector = dlib.get_frontal_face_detector()

    # init the predictor
    predictor = dlib.shape_predictor(config.face_landmark_path)

    # index when we will take the frame and get the emotions from the picture
    index = 1

    # getting from the mmo the info we need for the window
    mmo_data, owner_uid, camera_id, store_id = mmo_handler.get_json_mmo(config.MMO_PATH)

    if owner_uid and camera_id and store_id:
        config.change_store_id(store_id)
        config.change_camera_id(camera_id)
        config.change_owner_uid(owner_uid)
        print("Got all parameters: store id, camera id, owner id")
    else:
        print("Missing parameters: [store id, camera id, owner id]")
        return 1

    while True:

        # getting all frames from cameras
        frames_per_camera = device_manager_class.get_frames_from_all_cameras(device_manager, number_of_devices)

        ret = True if len(frames_per_camera) > 0 else False

        # we are still getting video from the camera
        if ret:
            for camera_frame in frames_per_camera:

                color_frame = camera_frame['color_frame']
                depth_frame = camera_frame['depth_frame']
                # getting the face rectangle from the frame
                face_rects = detector(color_frame, 0)

                # if we found some face/s in the frame
                if len(face_rects) > 0:

                    # start to analyse the face/s in the frame
                    for face in face_rects:
                        shape = predictor(color_frame, face)
                        shape = face_utils.shape_to_np(shape)

                        # estimate the head pose of the specific face
                        reprojectdst, euler_angle = helper_functions.get_head_pose(shape)

                        # init vars for measuring distance to object
                        dis_start = 0
                        dis_middle = 0
                        dis_end = 0

                        # getting the middle x,y of the shape detected
                        start_x, start_y = shape[0][0], shape[0][1]
                        middle_x, middle_y = shape[int(len(shape) / 2)][0], shape[int(len(shape) / 2)][1]
                        end_x, end_y = shape[-1][0], shape[-1][1]

                        if config.DEEP_DEBUG:
                            print("start x, y: {} , {}".format(start_x, start_y))
                            print("middle x, y: {} , {}".format(middle_x, middle_y))
                            print("end x, y: {} , {}".format(end_x, end_y))

                        # draw points for the face
                        for (x, y) in shape:

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

                        diss = helper_functions.normal_distances(diss, x_angle, y_angle)

                        # return value of too far objects is 0
                        # if no object detected because he is too far -> will put 1 as distance
                        distance_to_object = diss if diss is not None else config.DISTANCE

                        if config.DEEP_DEBUG:
                            print("distance to object: {}".format(distance_to_object))

                        # determine if the person wa looking to the left or right, up or down
                        left_right = "RIGHT" if x_angle >= 0 else "LEFT"
                        up_down = "UP" if y_angle <= 0 else "DOWN"

                        # getting the distance from camera for each axe
                        x_distances_camera_object = helper_functions.get_axe_distance_to_object(distance_to_object, x_angle)
                        y_distances_camera_object = helper_functions.get_axe_distance_to_object(distance_to_object, y_angle)

                        # getting the distance from camera to object
                        camera_object_distances = []
                        index_for_diss = 0
                        for x_dis in x_distances_camera_object:
                            camera_object_distance = helper_functions.calculate_distance(x_dis, y_distances_camera_object[index_for_diss])
                            camera_object_distances.append(camera_object_distance)
                            index_for_diss += 1

                        # check if the distance and direction fit any of the model object
                        found_object = mmo_handler.fit_model_object(mmo_data, camera_object_distances, left_right, up_down)

                        if found_object != "":
                            # means we found an object that the observer was looking at
                            index += 1

                            frame_timestamp = datetime.datetime.now().timestamp()
                            object_to_post = {
                                "owner_uid": config.OWNER_UID,
                                "store_id": config.STORE_ID,
                                "camera_id": config.CAMERA_ID,
                                "object_id": found_object,
                                "timestamp": frame_timestamp
                            }

                            if config.DEMO_MODE:
                                config.add_window_item(found_object, left_right, up_down)
                            else:
                                # saving the picture with the timestamp + object id
                                if index % config.FRAME_INTERVAL == 0:
                                    helper_functions.save_frame_as_picture(color_frame, found_object, frame_timestamp)

                                # posting the images(optional) and objects data to s3 and kinesis
                                kinesis_handler.start_posting(json.dumps(object_to_post),   # object
                                                              config.path_for_pictures)     # path to the pictures

                                # clear the image folder after posting -> only if we saved a picture
                                if index % config.FRAME_INTERVAL == 0:
                                    helper_functions.clear_images_folder()

                        else:
                            if config.DEEP_DEBUG:
                                print("no object found with distance from camera: {}".format(camera_object_distances))

                if config.DEMO_MODE:
                    print(json.dumps(config.items_in_window, indent=4))
                    if config.items_in_window.keys():

                        for item in config.items_in_window.keys():
                            if config.items_in_window[item]['position'] == "RIGHT.UP":
                                helper_functions.put_text(frame=color_frame, item=str(item), value=config.items_in_window[item]['value'],
                                         upper_position=(400, 50), lower_position=(470, 80), color=(0, 0, 205))

                            if config.items_in_window[item]['position'] == "RIGHT.DOWN":
                                helper_functions.put_text(frame=color_frame, item=str(item), value=config.items_in_window[item]['value'],
                                         upper_position=(400, 420), lower_position=(470, 440), color=(153, 153, 0))

                            if config.items_in_window[item]['position'] == "LEFT.UP":
                                helper_functions.put_text(frame=color_frame, item=str(item), value=config.items_in_window[item]['value'],
                                         upper_position=(50, 50), lower_position=(120, 80), color=(139, 0, 0))

                            if config.items_in_window[item]['position'] == "LEFT.DOWN":
                                helper_functions.put_text(frame=color_frame, item=str(item), value=config.items_in_window[item]['value'],
                                         upper_position=(50, 420), lower_position=(120, 440), color=(46, 139, 87))

                    cv2.namedWindow(camera_frame['device'], cv2.WND_PROP_ASPECT_RATIO)
                    cv2.setWindowProperty(camera_frame['device'], cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_NORMAL)
                    cv2.imshow(camera_frame['device'], color_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        device_manager.disable_streams()
                        break

            index += 1


if __name__ == '__main__':
    main()

