# coding=utf-8
# ------------------------------------------------------------
# v888_drop.py — YOLOv8 RKNN 实时检测 & ROS 发布
# ------------------------------------------------------------
import os, sys, cv2, time, queue, threading
import numpy as np
from rknnlite.api import RKNNLite
from collections import deque
import rospy
import logging

# 配置日志，防止冲突
os.environ['ROSCONSOLE_CONFIG_FILE']  = '/opt/ros/noetic/share/ros/config/rosconsole.config'
# 手动定义缺失的日志级别
if 'DEBUG' not in logging._nameToLevel:
    logging.addLevelName(logging.DEBUG,  'DEBUG')
if 'INFO' not in logging._nameToLevel:
    logging.addLevelName(logging.INFO,  'INFO')
# ================ 全局参数 ================
# 【注意】修改为绝对路径
MODEL_PATH = '/home/orangepi/Desktop/rknn_yolov8-ros1-main/multi_threaded-python-v8_ros_pub/model/123.rknn'
IMG_SIZE = (640,640)
OBJ_THRESH = 0.45   # 稍微提高阈值，减少误报
NMS_THRESH = 0.45

# 对应关系: 0:car, 5:tank (普通目标), 6:red (特殊目标)
CLASSES = ("car", "blockhouse", "H", "bridge", "tent", "tank", "redten")

# ================ RKNN处理工具函数 (保持原样) ================
def dfl(pos):
    n, c, h, w = pos.shape
    pos = pos.reshape(n, 4, c // 4, h, w)
    max_val = np.max(pos, axis=2, keepdims=True)
    exp_pos = np.exp(pos - max_val)
    softmax = exp_pos / np.sum(exp_pos, axis=2, keepdims=True)
    acc = np.arange(c // 4, dtype=np.float32).reshape(1, 1, -1, 1, 1)
    return np.sum(softmax * acc, axis=2)

def box_process(pos):
    gh, gw = pos.shape[2:4]
    col, row = np.meshgrid(np.arange(gw), np.arange(gh))
    grid = np.stack((col, row), 0).reshape(1, 2, gh, gw)
    stride = np.array([IMG_SIZE[1] // gw, IMG_SIZE[0] // gh]).reshape(1, 2, 1, 1)
    pos = dfl(pos)
    xy1 = grid + 0.5 - pos[:, :2]
    xy2 = grid + 0.5 + pos[:, 2:4]
    return np.concatenate((xy1 * stride, xy2 * stride), 1)

def nms_xyxy(boxes, scores):
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter)
        order = order[1 + np.where(iou <= NMS_THRESH)[0]]
    return keep

def filter_and_nms(boxes, cconf, oconf):
    cls_max = cconf.max(-1)
    cls_ids = cconf.argmax(-1)
    mask = cls_max * oconf.reshape(-1) >= OBJ_THRESH
    if not mask.any():
        return None, None, None
    boxes, scores, cls_ids = boxes[mask], (cls_max * oconf.reshape(-1))[mask], cls_ids[mask]
    fb, fs, fc = [], [], []
    for cid in np.unique(cls_ids):
        idx = np.where(cls_ids == cid)[0]
        keep = nms_xyxy(boxes[idx], scores[idx])
        fb.append(boxes[idx][keep])
        fs.append(scores[idx][keep])
        fc.append(np.full(len(keep), cid))
    return np.concatenate(fb), np.concatenate(fc), np.concatenate(fs)

def letter_box(img, new_shape=IMG_SIZE):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - nw) // 2, (new_shape[0] - nh) // 2
    img = cv2.resize(img, (nw, nh), cv2.INTER_LINEAR)
    canvas = np.zeros((new_shape[0], new_shape[1], 3), dtype=np.uint8)
    canvas[dh:dh+nh, dw:dw+nw] = img
    return canvas, r, (dw, dh)

def scale_boxes(boxes, src_shape, dw, dh, r):
    b = boxes.copy()
    b[:, [0,2]] = (b[:, [0,2]] - dw) / r
    b[:, [1,3]] = (b[:, [1,3]] - dh) / r
    h, w = src_shape
    b[:, [0,2]] = b[:, [0,2]].clip(0, w)
    b[:, [1,3]] = b[:, [1,3]].clip(0, h)
    return b

def get_class_name(cls_id):
    try:
        return CLASSES[cls_id]
    except IndexError:
        return f"class_{cls_id}"

# ================ 推理线程 ================
class InferenceWorker(threading.Thread):
    CORE_MAP = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1, 2: RKNNLite.NPU_CORE_2}
    def __init__(self, idx, model_path, in_q, out_q):
        super().__init__(daemon=True)
        self.in_q, self.out_q = in_q, out_q
        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(model_path) != 0:
            print("Load RKNN failed!")
            sys.exit(1)
        if self.rknn.init_runtime(core_mask=self.CORE_MAP[idx]) != 0:
            print("Init runtime failed!")
            sys.exit(1)
        print(f'[Worker-{idx}] init OK')

    def run(self):
        while True:
            fid, frame = self.in_q.get()
            if fid is None: break
            h0, w0 = frame.shape[:2]
            img, r, (dw, dh) = letter_box(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = self.rknn.inference([np.expand_dims(img, 0)])

            # YOLOv8 输出处理
            boxes, cconfs, oconfs = [], [], []
            pair = len(out) // 3
            for i in range(3):
                boxes.append(box_process(out[pair*i]))
                cconfs.append(out[pair*i+1])
                oconfs.append(np.ones_like(out[pair*i+1][:, :1, :, :], np.float32))

            # 合并结果
            merge = lambda xs: np.concatenate([x.transpose(0,2,3,1).reshape(-1, x.shape[1]) for x in xs])
            b, cls, s = filter_and_nms(merge(boxes), merge(cconfs), merge(oconfs))
            if b is not None:
                b = scale_boxes(b, (h0, w0), dw, dh, r)
            self.out_q.put((fid, b, cls, s))
        self.rknn.release()

# ================ 主程序 ================
def main():
    rospy.init_node("yolov8_detection_node")

    # 初始化队列与线程
    in_qs = [queue.Queue(1) for _ in range(3)]
    out_q = queue.Queue(3)
    workers = [InferenceWorker(i, MODEL_PATH, in_qs[i], out_q) for i in range(3)]
    for w in workers: w.start()

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 640, 640)

    # 清理参数
    for p in ["center_x", "center_y", "cls_ids", "scores", "ready", "center_x_list", "center_y_list"]:
        if rospy.has_param(p): rospy.delete_param(p)
    rospy.set_param("ready", False)

    fid = 0
    history = deque(maxlen=5)

    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret: break

            # 发送推理任务
            target_worker = fid % 3
            if in_qs[target_worker].empty():
                in_qs[target_worker].put((fid, frame.copy()))
            fid += 1

            # 获取推理结果
            boxes, cls_ids, scores = None, None, None
            while not out_q.empty():
                _, b, c, s = out_q.get()
                if b is not None:
                    boxes, cls_ids, scores = b, c, s

            if boxes is not None:
                # 绘制所有目标
                for box, cls, sc in zip(boxes, cls_ids, scores):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f'{get_class_name(cls)} {sc:.2f}', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # 计算所有目标的中心点列表
                center_x_list = []
                center_y_list = []
                for box in boxes:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    center_x_list.append(float(cx))
                    center_y_list.append(float(cy))

                # 取最高得分目标的中心（用于向下兼容）
                best_idx = np.argmax(scores)
                bx = (boxes[best_idx][0] + boxes[best_idx][2]) / 2
                by = (boxes[best_idx][1] + boxes[best_idx][3]) / 2

                # 发布参数
                rospy.set_param("cls_ids", cls_ids.tolist())
                rospy.set_param("scores", scores.tolist())
                rospy.set_param("center_x", float(bx))
                rospy.set_param("center_y", float(by))
                rospy.set_param("center_x_list", center_x_list)   # 新增：所有目标的中心x
                rospy.set_param("center_y_list", center_y_list)   # 新增：所有目标的中心y
                rospy.set_param("ready", True)
                #123.py
                rospy.set_param("drop_flag", True)   # 增加这一行
            else:
                rospy.set_param("ready", False)

            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) == ord('q'): break
            rospy.sleep(0.01)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for q in in_qs: q.put((None, None))
        for w in workers: w.join()

if __name__ == '__main__':
    main()
