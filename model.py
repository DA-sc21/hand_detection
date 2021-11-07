import cv2
from PIL import Image
import numpy as np
import os
#import tensorflow as tf

from modules.yolo import YOLO
from modules import ssd_utils as ssd_utils
import mediapipe as mp

class Model():

    def __init__(self, opt):
        self.opt = opt

        if opt.ObjectDetection == 'ssdmobilenetv1':
            MODEL_NAME = 'modules/models/ssd_mobilenetv1'
            self.ObjectDetection, self.sess = ssd_utils.load_inference_graph(opt.ObjectDetection)
        elif opt.ObjectDetection == 'ssdmobilenetv2':
            MODEL_NAME = 'modules/models/ssd_mobilenetv2'
            self.ObjectDetection, self.sess = ssd_utils.load_inference_graph(opt.ObjectDetection)
        elif opt.ObjectDetection == 'yolov4-tiny':
            self.ObjectDetection = YOLO("modules/models/yolov4-tiny/yolov4-tiny-custom.cfg", "modules/models/yolov4-tiny/yolov4-tiny-custom_best.weights", ["hand"])
        elif opt.ObjectDetection == 'yolov3-tiny':
            self.ObjectDetection = YOLO("modules/models/yolov3-tiny/yolov3-tiny-prn-custom.cfg", "modules/models/yolov3-tiny/yolov3-tiny-prn-custom.weights", ["hand"])
        elif opt.ObjectDetection == 'mediapipe':
            self.ObjectDetection = mp.solutions
        else :
            raise Exception('No ObjectDetection module specified')
    
    def forward(self,model,files,show):
        if show:
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

            elif model == 'ssdmobilenetv1' or model == 'ssdmobilenetv2':
                for file in files:
                    print(file)
                    mat = cv2.imread(file)
                    num_hands_detect = 4

                    im_height , im_width = mat.shape[:2]
                    boxes, scores = ssd_utils.detect_objects(mat,
                                                      self.ObjectDetection, self.sess)

                    # draw bounding boxes on frame
                    ssd_utils.draw_box_on_image(num_hands_detect, 0.2,
                                             scores, boxes, im_width, im_height,
                                             mat)
                    cv2.imshow(model,
                           cv2.cvtColor(mat, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(0)
                    print(len(boxes))

                cv2.destroyAllWindows()

            elif model == 'mediapipe':
                for file in files:
                    hand_img = Image.open(file)

                    with self.ObjectDetection.hands.Hands(
                          static_image_mode=True,
                          max_num_hands=4,
                          min_detection_confidence=0.3) as hands:
                        image = cv2.cvtColor(np.array(hand_img), cv2.COLOR_RGB2BGR)
                        # Convert the BGR image to RGB before processing.
                        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        print("hand_num : ", len(results.multi_handedness))

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                self.ObjectDetection.drawing_utils.draw_landmarks(
                                    image,
                                    hand_landmarks,
                                    self.ObjectDetection.hands.HAND_CONNECTIONS,
                                    self.ObjectDetection.drawing_styles.get_default_hand_landmarks_style(),
                                    self.ObjectDetection.drawing_styles.get_default_hand_connections_style())
                            # Flip the image horizontally for a selfie-view display.
                            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                            cv2.waitKey(0)
                    cv2.destroyAllWindows()

        else :
            if model == 'yolov4-tiny' or model == 'yolov3-tiny':
                conf_sum = 0
                detection_count = 0
                for file in files:
                    print(file)
                    mat = cv2.imread(file)
                    width, height, inference_time, results = self.ObjectDetection.inference(mat)
                    print(results)
                    print("%s in %s seconds: %s classes found!" %
                  (os.path.basename(file), round(inference_time, 2), len(results)))

                print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))

            elif model == 'ssdmobilenetv1' or model == 'ssdmobilenetv2':
                for file in files:
                    mat = cv2.imread(file)
                    num_hands_detect = 4
                    im_height , im_width = mat.shape[:2]
                    boxes, scores = ssd_utils.detect_objects(mat,
                                                      self.ObjectDetection, self.sess)

                    print(boxes)

            elif model == 'mediapipe':
                for file in files:
                    hand_img = Image.open(file)

                    with self.ObjectDetection.hands.Hands(
                          static_image_mode=True,
                          max_num_hands=4,
                          min_detection_confidence=0.3) as hands:
                        image = cv2.cvtColor(np.array(hand_img), cv2.COLOR_RGB2BGR)
                        # Convert the BGR image to RGB before processing.
                        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        print(results.multi_handedness)
                        print("hand_num : ", len(results.multi_handedness))

        