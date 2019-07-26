import cv2
import numpy as np


class ObjectDetector:
    def __init__(self, streamURL=0, confidence=0.6):
        # Create a VideoCapture object and rad form WebCam
        self.vcap = cv2.VideoCapture(streamURL)
        self.confidence = confidence
        # caffemodel and deloy 
        self.prototxt = "deploy.prototxt"
        self.caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
        print('[INFO] loading  a caffe model form given location ')

        self.dnnet = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
        # Check if Webcam is successfully opened
        if self.vcap.isOpened() is False:
            print("Error in opening the video stream....")

    def draw_bounding_box(self, frame, box):
        # Object location in Frame

        # Bounding box Co-ordinates
        x, y, x_plush_h, y_plush_w = box.astype('int')
        # print(x, y, x_plush_h, y_plush_w)
        cv2.rectangle(frame, (x, y,), (x_plush_h, y_plush_w), (255, 255, 0), 2)
        cv2.imshow('Object Detected ', frame)
        out = cv2.VideoWriter('output.mp4', -1, 20.0, (640, 480))
        # frame = cv2.flip(frame, 1)
        # out.write(frame)


def detect_object(self, frame):
    # Get  the height and Width of frame
    (Height, Width) = frame.shape[:2]
    imgblob = cv2.dnn.blobFromImage(frame, 1, (Height, Width), (104.0, 177.0, 123.0))
    self.dnnet.setInput(imgblob)
    detectedOutputs = self.dnnet.forward()

    # For get the class and location of object detected,
    # There is a fix index for class, location and confidence    value in @detectedOutputs array .
    for i in range(0, detectedOutputs.shape[2]):
        confidence = detectedOutputs[0, 0, i, 2]  # Prediction Confidence
        # If predicted confidence is higher or equal to the given threshold confidence
        # get the bounding box co-ordinates
        if confidence >= self.confidence:
            # objectClass = detectedOutputs[0, 0, i, 1]
            # print("Class of object", objectClass)
            box = detectedOutputs[0, 0, i, 3:7] * np.array([Width, Height, Width, Height])
            self.draw_bounding_box(frame, box)


def readframe(self):
    while self.vcap.isOpened():
        # Read Video Stream frame by frame
        rect, frame = self.vcap.read()
        height, width = frame.shape[:2]
        # print(height, width)
        resize = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        height, width = resize.shape[:2]
        if rect is True:
            # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
            self.detect_object(frame)
            # Press q key to stop the video frame stream
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # After completing desire work stop the video steam
    self.vcap.release()
    # Destroy all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    otrack = ObjectDetector()
    otrack.readframe()
