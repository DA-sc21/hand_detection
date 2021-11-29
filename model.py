import cv2
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import time

from modules.yolo import YOLO
from modules import ssd_utils as ssd_utils
import mediapipe as mp

class Model():

    def __init__(self, opt):
        self.opt = opt

        if opt.ObjectDetection == 'ssdmobilenetv1':
            self.ObjectDetection, self.sess = ssd_utils.load_inference_graph(opt.ObjectDetection)
        elif opt.ObjectDetection == 'ssdmobilenetv2':
            self.ObjectDetection, self.sess = ssd_utils.load_inference_graph(opt.ObjectDetection)
        elif opt.ObjectDetection == 'yolov4-tiny':
            if opt.mode == "1_0_0":
                self.ObjectDetection = YOLO("/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov4-tiny/yolov4-tiny-custom.cfg", "/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov4-tiny/yolov4-tiny-custom_only_egodataset.weights", ["hand"])
            elif opt.mode == "2_1_2":
                pass
            elif opt.mode == "4_1_4":
                self.ObjectDetection = YOLO("/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov4-tiny/yolov4-tiny-custom.cfg", "/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov4-tiny/yolov4-tiny-custom_4_1_4.weights", ["hand"])
            else :
                self.ObjectDetection = YOLO("/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov4-tiny/yolov4-tiny-custom.cfg", "/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov4-tiny/yolov4-tiny-custom_8_1_4.weights", ["hand"])
        elif opt.ObjectDetection == 'yolov3-tiny':
            self.ObjectDetection = YOLO("/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov3-tiny/yolov3-tiny-prn-custom.cfg", "/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov3-tiny/yolov3-tiny-prn-custom_only_egodataset.weights", ["hand"])
            # self.ObjectDetection = YOLO("/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov3-tiny/cross-hands-tiny-prn.cfg", "/Users/yejoonko/git/Project/Capstone/hand_detection/modules/models/yolov3-tiny/cross-hands-tiny-prn.weights", ["hand"])
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
                        cv2.rectangle(mat, (x, y), (x + w, y + h), color, 3)
                        text = "%s (%s)" % (name, round(confidence, 2))
                        cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, color, 1)

                        print("%s with %s confidence" % (name, round(confidence, 2)))

                        # cv2.imwrite("export.jpg", mat)

                    # show the output image\
                    cv2.imshow('image',mat)
                    cv2.waitKey(0)

                print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
                cv2.destroyAllWindows()

            elif model == 'ssdmobilenetv1' or model == 'ssdmobilenetv2':
                for file in files:
                    print(file)
                    img = Image.open(file)
                    image_np = np.array(img)
                    num_hands_detect = 4

                    im_width, im_height = img.size
                    boxes, scores,inference_time = ssd_utils.detect_objects(image_np,
                                                      self.ObjectDetection, self.sess)

                    # draw bounding boxes on frame
                    ssd_utils.draw_box_on_image(num_hands_detect, 0.3,
                                             scores, boxes, im_width, im_height,
                                             image_np)
                    cv2.imshow(model,
                           cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
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
                        try :
                            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            print("hand_num : ", len(results.multi_handedness))
                        except :
                            print("no hand")
                            continue

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

                detection_result = [] #file_path(확장자 제거),'hand',confidence, x_min, y_min, x_max, y_max
                mean_inference_time = 0
                for file in files:
                    file_path = file.split("/")[-1][:-4]
                    mat = cv2.imread(file)
                    width, height, inference_time, results = self.ObjectDetection.inference(mat)
                    mean_inference_time += inference_time
                    for result in results:
                        id, name, conf, x, y, w, h = result
                        detection_result.append([file_path, name, str(conf),str(x),str(y),str(x+w),str(y+h)])
                    print("%s in %s seconds: %s classes found!" %
                  (os.path.basename(file), round(inference_time, 2), len(results)))
                # print(detection_result)
                # print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
                mean_inference_time /= len(files)
                return detection_result, mean_inference_time

            elif model == 'ssdmobilenetv1' or model == 'ssdmobilenetv2':
                mean_inference_time = 0
                detection_result = [] #file_path(확장자 제거),'hand',confidence, x_min, y_min, x_max, y_max
                for file in files:
                    img = Image.open(file)
                    image_np = np.array(img)
                    num_hands_detect = 4

                    im_width, im_height = img.size
                    boxes, scores,inference_time = ssd_utils.detect_objects(image_np,
                                                      self.ObjectDetection, self.sess)


                    file_path = file.split("/")[-1][:-4]

                    mean_inference_time += inference_time
                    score_thresh = 0.3
                    for i in range(len(boxes)):
                        if scores[i] > score_thresh:
                            left, right, top, bottom = boxes[i][1] * im_width, boxes[i][3] * im_width, boxes[i][0] * im_height, boxes[i][2] * im_height
                            detection_result.append([file_path, 'hand', str(scores[i]),str(left),
                            str(top),str(right),str(bottom)])

                    #print(detection_result)

                mean_inference_time /= len(files)
                return detection_result, mean_inference_time

            elif model == 'mediapipe':
                mean_inference_time = 0
                for file in files:
                    hand_img = Image.open(file)

                    with self.ObjectDetection.hands.Hands(
                          static_image_mode=True,
                          max_num_hands=4,
                          min_detection_confidence=0.3) as hands:
                        image = cv2.cvtColor(np.array(hand_img), cv2.COLOR_RGB2BGR)
                        # Convert the BGR image to RGB before processing.

                        try :
                            start = time.time()
                            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            end = time.time()
                            print("hand_num : ", len(results.multi_handedness))
                        except :
                            print("no hand")
                            continue
                        mean_inference_time += (end-start)
                        print(results.multi_handedness)
                        print("hand_num : ", len(results.multi_handedness))

                mean_inference_time /= len(files)

                return [],mean_inference_time

        
