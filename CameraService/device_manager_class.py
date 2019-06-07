import pyrealsense2 as rs
import re
import numpy as np


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
    def __init__(self, pipeline_configuration):
        """
        Class to manage the Intel RealSense devices

        Parameters:
        -----------
        context 	: rs.context()
                                     The context created for using the realsense library
        pipeline_configuration 	: rs.config()
                                   The realsense library configuration to be used for the application

        """

        assert isinstance(pipeline_configuration, type(rs.config()))
        self._context = rs.context()
        self._available_devices = enumerate_connected_devices(self._context)
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

        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.exposure, 8400.0)

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
        color_frame = frames.get_color_frame()

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

        # remove the background from 1.5 meters and on
        color_frame = clip_frame_by_distance(device_manager.get_pipeline, color_frame, depth_frame, 1.5)

        frames.append({"device": str(serial), "color_frame": color_frame, "depth_frame": depth_frame})

    return frames


def clip_frame_by_distance(profile, color_frame, depth_frame, distance=1.5):

    # getting the scale from the sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # adjust the clipping distance
    clipping_distance = distance / depth_scale

    # convert the frames into np arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack(
        (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    return bg_removed

