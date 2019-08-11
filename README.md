[![Build Status](https://travis-ci.com/everythingisdata/ObjectTracking.svg?branch=master)](https://travis-ci.com/everythingisdata/ObjectTracking)
# Object Detection
---
Object Detection is Deep learning problem. It has three sections (Object Recognition, Object localization by bound box, object identification). Initially, we will detect an localize the object. 
 - Object Recognition
- Object Localization
 - Object Tracking(TODO)
 
First, we will use an object recognition algorithm. There are a couple of algorithms and pre-train model. Which can help be used for object recognition?
- HAAR
- YOLO (You Only Look Once)
- SSD (Single-Shot Detection)

In this, I have used SSD pre-train Caffe model for object recognition.
 - Tracking is the process of tracking the co-ordinates of moving object.

Pre-requisite.
 - Python 3.5
 - OpenCV
 - Pre-train model like Caffe or TensorFlow
