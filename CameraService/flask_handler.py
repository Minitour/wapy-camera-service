from flask import Flask
import subprocess

app = Flask(__name__)


@app.route('/start_camera_service', methods=['GET'])
def start():
    print("starting the camera service")
    command = "nssm start camera-service"
    change_mod(command)


@app.route('/stop_camera_service', methods=['GET'])
def stop():
    print("stopping the camera service")
    command = "nssm stop camera-service"
    change_mod(command)


def change_mod(command):

    MyOut = subprocess.Popen(str(command).split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
    stdout,stderr = MyOut.communicate()
    print(stdout)
    print(stderr)


if __name__ == "__main__":
    app.run()