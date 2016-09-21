# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2



def capture(camera, rawCapture):
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the frame, then initialize the timestamp
        # and occupied/unoccupied text
        frame = frame.array
        yield frame

def find_faces(frame, classifier, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    return faces

def log(faces):
    print(faces)

def highlight_faces(frame, faces, rgb=(0,255,0), width=1):
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), rgb, width)

def show_frame(frame):
    # show the frame
    cv2.imshow("Frame", frame)


def run():
    # initialize the camera and grab a reference to the raw camera capture
    
    resolution = (320, 240)
    
    camera = PiCamera()
    camera.resolution = resolution
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=resolution)
     
    # allow the camera to warmup
    time.sleep(0.1)

    definitions = ['classifiers/haarcascade_frontalface_default.xml', 'classifiers/haarcascade_profileface.xml', 'classifiers/lbpcascade_profileface.xml']

    classifiers = [cv2.CascadeClassifier(definition) for definition in definitions]
    for frame in capture(camera, rawCapture):
#       rimg=frame.copy()
#       rimg=cv2.flip(frame,1)
        for classifier in classifiers:
            faces = find_faces(frame, classifier)
            log(faces)
            highlight_faces(frame, faces)

        show_frame(frame)
        key = cv2.waitKey(1) & 0xFF
 
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

run()
