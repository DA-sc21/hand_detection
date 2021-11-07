import cv2
from PIL import Image
import numpy as np
import os

from modules.yolo import YOLO
import mediapipe as mp

class Model():

    def __init__(self, opt):
        self.opt = opt

        if opt.ObjectDetection == 'ssdMobileNetv1':
            #self.ObjectDetection = ssdMobileNetv1()
            pass
        elif opt.ObjectDetection == 'ssdMobileNetv2':
            pass
        elif opt.ObjectDetection == 'yolov4-tiny':
            self.ObjectDetection = YOLO("modules/models/yolov4-tiny-custom.cfg", "modules/models/yolov4-tiny-custom_best.weights", ["hand"])
        elif opt.ObjectDetection == 'yolov3-tiny':
            self.ObjectDetection = YOLO("modules/models/yolov3-tiny-prn-custom.cfg", "modules/models/yolov3-tiny-prn-custom.weights", ["hand"])
        elif opt.ObjectDetection == 'mediapipe':
            self.ObjectDetection = mp.solutions.hands
        else :
            raise Exception('No ObjectDetection module specified')
    
    def forward(self,model,files):
        if model == 'yolov4-tiny' or model == 'yolov3-tiny':
            conf_sum = 0
            detection_count = 0
            print(files)
            for file in files:
                print(file)
                mat = cv2.imread(file)

                width, height, inference_time, results = self.ObjectDetection.inference(mat)
                print(results)

                print("%s in %s seconds: %s classes found!" %
              (os.path.basename(file), round(inference_time, 2), len(results)))

                output = []

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 848, 640)

                for detection in results:
                    id, name, confidence, x, y, w, h = detection
                    cx = x + (w / 2)
                    cy = y + (h / 2)

                    conf_sum += confidence
                    detection_count += 1

                    # draw a bounding box rectangle and label on the image
                    color = (255, 0, 255)
                    cv2.rectangle(mat, (x, y), (x + w, y + h), color, 1)
                    text = "%s (%s)" % (name, round(confidence, 2))
                    cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, color, 1)

                    print("%s with %s confidence" % (name, round(confidence, 2)))

                    # cv2.imwrite("export.jpg", mat)

                # show the output image
                cv2.imshow('image', mat)
                cv2.waitKey(0)

            print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
            cv2.destroyAllWindows()

        elif model == 'ssdMobileNetv1' or model == 'ssdMobileNetv1':
            pass

        elif model == 'mediapipe':
            for file in files:
                hand_img = Image.open(file)

                with self.ObjectDetection.Hands(
                      static_image_mode=True,
                      max_num_hands=4,
                      min_detection_confidence=0.3) as hands:
                    image = cv2.cvtColor(np.array(hand_img), cv2.COLOR_RGB2BGR)
                    # Convert the BGR image to RGB before processing.
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    print(results.multi_handedness)
                    print("hand_num : ", len(results.multi_handedness))
                    

    