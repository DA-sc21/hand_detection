import argparse
import glob
import os
import cv2

from model import Model
import evaluate_mAP

def test(opt):
    labels = {'hand' : 0}

    model = Model(opt)
    print('model input parameters : ', opt.ObjectDetection)

    files = sorted(glob.glob("%s/*.jpg" % opt.eval_data))

    results,mean_inference_time = model.forward(opt.ObjectDetection,files,False)
    if opt.ObjectDetection == 'mediapipe':
        print("{model} average Inference time {time}".format(model=opt.ObjectDetection,time=mean_inference_time))
    else :
        precision=evaluate_mAP.evaluate(opt.ObjectDetection,'test',results,opt.eval_data)
        print("{model} average Inference time {time}".format(model=opt.ObjectDetection,time=mean_inference_time))
        print("{model} average Precision {precision}".format(model=opt.ObjectDetection,precision=precision))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    # parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    # parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Model Architecture """
    parser.add_argument('--ObjectDetection', type=str, required=True, help='ObjectDetection stage. ssdmobilenetv1|ssdmobilenetv2 \
    yolov3-tiny|yolov4-tiny|mediapipe')

    opt = parser.parse_args()

    test(opt)