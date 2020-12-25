import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as trans
import PIL.Image as Image
import json
import time

rafdb_table = {1:"Surprise", 2:"Fear", 3:"Disgust", 4:"Happiness", 5:"Sadness", 6:"Anger", 7:"Neutral"}
affectnet_table = {0:"Happy", 1:"Sad", 2:"Surprise", 3:"Fear", 4:"Disgust", 5:"Anger"}
affectnet7_table = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger"}
action_table = {0: 'Phone', 1: 'Yawn', 2: 'Smoke', 3: 'Normal'}

class CV2FaceDetector:
    def __init__(self, model_para_dir='ckpt/haarcascade_frontalface_default.xml'):
        self.detector = cv2.CascadeClassifier(model_para_dir)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, minSize=(100, 100))
        return faces

def get_trained_model(model, param_path, map_location="cpu"):
    ckpt = torch.load(param_path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

class FaceBatchSolver:
    def __init__(self, detector, classifier, device, label, batch_size=128, DEBUG=False):
        self.detector = detector
        self.classifier = classifier
        self.device = device
        #self.solver = solver
        self.batch_size = batch_size
        self.resizer = trans.Compose([
            trans.Resize((224)),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        self.label_idx = [k for k,v in label.items()]

        self.classifier.to(device)

        self.DEBUG = DEBUG

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def solve(self, imgs):

        if self.DEBUG:
            start_t = time.time()

        if len(imgs) != self.batch_size:
            raise Exception("batch size not match. batch:%d imgs:%d"%(self.batch_size, len(imgs)))
        input = torch.zeros((self.batch_size, 3, 224, 224))
        boxes_vector = np.zeros((self.batch_size, 4))
        exist_vector = np.zeros((self.batch_size), dtype=np.bool)
        index_vector = np.zeros((self.batch_size))
        for i,(idx,img) in enumerate(imgs):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect(img)
            index_vector[i] = idx
            if len(faces) != 0:
                x,y,w,h = faces[0]
                crop = img[y:y+h, x:x+w, :]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                #print(crop.shape)
                crop = self.resizer(Image.fromarray(crop))
                #print(crop.shape)
                input[i, ...] = crop
                boxes_vector[i, :] = np.array([x,y,w,h])
                exist_vector[i] = True
            else:
                exist_vector[i] = False

        if self.DEBUG:
            now_t = time.time()
            print("detect time:%d s"%(now_t - start_t))
            start_t = now_t

        #input = torch.Tensor(input)
        input.to(self.device)
        output = self.classifier(input).relu()[:, self.label_idx, ...]
        output = output.softmax(1)
        _, cls = torch.max(output, 1)
        cls = [x for x in cls.detach().numpy().tolist()]
        prob_vector = output.detach().numpy().tolist()
        prob_vector = np.round(prob_vector, 3)

        if self.DEBUG:
            now_t = time.time()
            print("classify time:%d s"%(now_t - start_t))
            start_t = now_t

        result = {
            "Vector":prob_vector.tolist(),
            "Class":cls,
            "Boxes":boxes_vector.astype(int).tolist(),
            "Exist":exist_vector.tolist(),
            "Index":index_vector.astype(int).tolist()
        }
        return result

    def solve_path(self, path):
        catcher = cv2.VideoCapture(path)
        if not catcher.isOpened():
            raise Exception("Video not found.")
        imgs = []
        for i in range(self.batch_size):
            _, img = catcher.read()
            imgs.append((i, img))
        return self.solve(imgs)

    def solve_in_json(self, imgs):
        return json.dumps(self.solve(imgs))

    @staticmethod
    def to_json(dct):
        return json.dumps(dct)

class ActionBatchSolver:
    # not implemented
    def __init__(self, classifier, device, label, batch_size):
        self.batch_size = batch_size
        self.device = device
        self.classifier = classifier
        self.resizer = trans.Compose([
            trans.Scale(512),
            trans.CenterCrop(448),
            # trans.ColorJitter(brightness=(1, 1)),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

        # not used
        self.action_table = label

    def solve(self, imgs):

        if len(imgs) != self.batch_size:
            raise Exception("batch size not match. batch:%d imgs:%d"%(self.batch_size, len(imgs)))
        input = torch.zeros((self.batch_size, 3, 448, 448))
        index_vector = np.zeros((self.batch_size))
        for i,(idx,img) in enumerate(imgs):
            index_vector[i] = idx
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.resizer(Image.fromarray(img))
            input[i, ...] = img

        self.classifier.to(self.device)
        input.to(self.device)

        pred = self.classifier(input).detach()
        print(pred)
        pred = F.softmax(pred, dim=1)
        print(pred)
        prob, res = torch.max(pred, 1)
        res = res.detach().numpy()
        prob = prob.detach().numpy()
        pred = pred.cpu().numpy()
        #if res == 0 and pred[int(res)] < 0.9:
        #     res = 3
        res[np.logical_and(res==0, pred[:, 0]<0.9)] = 3

        cls = np.eye(4, dtype=np.bool)[res]
        return {"Vector": pred.astype(float).tolist(),
                "Class": cls.astype(int).tolist(),
                "Index": index_vector.astype(int).tolist()
                }

    def solve_path(self, path):
        catcher = cv2.VideoCapture(path)
        if not catcher.isOpened():
            raise Exception("Video not found.")
        imgs = []
        for i in range(self.batch_size):
            _, img = catcher.read()
            imgs.append((i, img))
        return self.solve(imgs)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def to_json(dct):
        return json.dumps(dct)

if __name__=="__main__":
    from solver import *

    detector = CV2FaceDetector()
    mbn = get_trained_model(mbnet.MobileNetV3_Small(), "ckpt/affectnet_mobilenetv3_small_acc83.pth.tar")
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    bsolver = BatchSolver(detector, mbn, device, affectnet_table, batch_size=32)
    def img_catcher(path, num):
        imgs = []
        vid = cv2.VideoCapture(path)
        for i in range(num):
            _, img = vid.read()
            imgs.append((i, img))
        return imgs
    string = bsolver.solve_in_json(img_catcher("./test/WeChat_20201120140410.mp4", 32))
    print(len(string))
    f = open("test/outputtest.json", "w")
    f.write(string)
    f.close()
