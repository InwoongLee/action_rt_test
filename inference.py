import torch
import numpy as np
import cv2
import time
torch.backends.cudnn.benchmark = True

import resnet_video_original as models

from torch2trt import torch2trt
from torch2trt import TRTModule

def resize_with_pad(image, height=640, width=480):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image 


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    # with open('./dataloaders/ucf_labels.txt', 'r') as f:
    # with open('./dataloaders/hmdb_labels.txt', 'r') as f:
    with open('./dataloaders/ntu_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model = models.r3d_18(num_classes=60)
    # checkpoint = torch.load('models/R3D-18-YOLOv3-ntu60_epoch-59.pth.tar', map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    x = torch.ones((1,3,16,112,112)).cuda()
    model_trt = torch2trt(model, [x])

    torch.save(model_trt.state_dict(), 'r3d_18_trt.pth')

    # read video
    video = 'samples/S002C003P011R001A001_rgb.avi' 
    cap = cv2.VideoCapture(video)
    retaining = True

    t0 = time.time()
    cnt = 0
    clip = []
    clip_size = 64
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])

        if cnt == 0:
            tmp_zero = np.zeros_like(tmp)
            for ii in range(clip_size-1):
                clip.append(tmp_zero)
        cnt += 1

        clip.append(tmp)

        frame = resize_with_pad(frame, 1280, 960)

        if len(clip) == clip_size:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs[:,:,0:clip_size:4,:,:])
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

            t = time.time()
            with torch.no_grad():
                outputs = model.forward(inputs)
            FPS_Action = 1/(time.time() - t)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            
            if probs[0][label] >0.7:
                cv2.putText(frame, "Pred: "+class_names[label].split(' ')[-1].strip() +
                             " %.4f" % probs[0][label], (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (53, 22, 233), 3)
            clip.pop(0)

        FPS_Total = cnt/(time.time() - t0)
        cv2.putText(frame, "FPS_Action: %.2f, FPS_Total: %.2f" % 
                    (FPS_Action, FPS_Total), (60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (53, 233, 22), 2)

        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









