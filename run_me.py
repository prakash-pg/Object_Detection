from detector import *


modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz"

classFile = "coco.names"
imagePath = "D:/object_detection/test/7.jpg"
videoPath = "D:/object_detection/test/video2.mp4"
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)
