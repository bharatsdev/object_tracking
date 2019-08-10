"""
This Python file for Object Tracking..

"""
import cv2
import numpy as np
import gc
from imutils.video import VideoStream

gc.enable()


class ObjectDetector:

    def __init__(self, streamURL=0, confidence=0.6):
        """
        :param streamURL: This can be any stream url (like camera), default=0 mean it will user you machine webcam
        :param confidence: This is a threshold 
        """
        print("[INFO] : Stream starting.....")
        self.vcap = cv2.VideoCapture(streamURL)
        self.confidence = confidence

        print("[INFO] : Loading SSD pre-train model.... ")
        self.prototxt = "deploy.prototxt"
        self.caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
        print("[INFO] : Loading SSD pre-train model done! ")

        # Validating Webcam is successfully opened
        if self.vcap.isOpened() is False:
            print("Error in opening the video stream....")

    def draw_bounding_box(self, frame, box):
        """
        :param frame: It is image frame 
        :param box:  It's contains coordinates of bounding box..
        """

        # Bounding box Co-ordinates
        startX, startY, endX, endY = box.astype('int')
        # print(x, y, x_plush_h, y_plush_w)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 2)
        cv2.imshow('Object Detected ', frame)
        # out = cv2.VideoWriter('output.mp4', -1, 20.0, (640, 480))
        # frame = cv2.flip(frame, 1)
        # out.write(frame)

    def detect_object(self, frame):
        """
        :rtype: object
        """
        # Get  the height and Width of frame
        (height, width) = frame.shape[:2]
        img_blob = cv2.dnn.blobFromImage(frame, 1, (height, width), (104.0, 177.0, 123.0))
        self.net.setInput(img_blob)

        detections = self.net.forward()

        # For get the class and location of object detected,
        # There is a fix index for class, location and confidence    value in @detectedOutputs array .
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Prediction Confidence
            # If predicted confidence is higher or equal to the given threshold confidence
            # get the bounding box co-ordinates
            if confidence >= self.confidence:
                # objectClass = detectedOutputs[0, 0, i, 1]
                # print("Class of object", objectClass)
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                self.draw_bounding_box(frame, box)

    def read_object_frames(self):
        while self.vcap.isOpened():
            # Read Video Stream frame by frame
            has_frame, frame = self.vcap.read()
            if has_frame:
                height, width = frame.shape[:2]
                # print(height, width)
                resize = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
                height, width = resize.shape[:2]
                self.detect_object(frame)
                # Press q key to stop the video frame stream
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        # After completing desire work stop the video steam
        self.vcap.release()
        # Destroy all the frames
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("[INFO] : Object tracking starting....")
    track = ObjectDetector()
    track.read_object_frames()