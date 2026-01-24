# coding=utf-8
# ------------------------------------------------------------
# rknn_v8_simplified.py — YOLOv8 RKNN 实时检测简化版
# ------------------------------------------------------------
import os, sys, cv2, time, queue, threading
import numpy as np
from rknnlite.api import RKNNLite
from collections import deque
import rospy
import logging 
import os

# 修改日志，防止rknn与rospy冲突
# 确保使用正确的配置文件 
os.environ['ROSCONSOLE_CONFIG_FILE']  = '/opt/ros/noetic/share/ros/config/rosconsole.config' 
# 手动定义缺失的日志级别
if 'DEBUG' not in logging._nameToLevel:
    logging.addLevelName(logging.DEBUG,  'DEBUG')
if 'INFO' not in logging._nameToLevel:
    logging.addLevelName(logging.INFO,  'INFO')

# ================ 全局参数 ================
MODEL_PATH = '/home/orangepi/Desktop/rknn_yolov8-ros1-main/multi_threaded-python-v8_ros_pub/model/1.rknn'   			# ← 改成你的 .rknn 路径
IMG_SIZE = (640, 640)                           # 推理输入分辨率
OBJ_THRESH = 0.25                               # 置信度阈值
NMS_THRESH = 0.5								# IoU-NMS 阈值
Flags = True                                 
# 根据实际模型修改类别，如果不知道可以先设多一些
CLASSES = ("car", "blockhouse","H","bridge","tent","tank")

# ================ 后处理工具函数 ================
def dfl(pos):
    n, c, h, w = pos.shape
    pos = pos.reshape(n, 4, c // 4, h, w)
    # 修复数值溢出问题：减去最大值再计算softmax
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

# 安全的类别名获取函数
def get_class_name(cls_id):
    """根据类别索引获取类别名"""
    try:
        return CLASSES[cls_id]
    except IndexError:
        return f"class_{cls_id}"

# ================ 推理线程 ================
class InferenceWorker(threading.Thread):
    CORE_MAP = {0: RKNNLite.NPU_CORE_0,
                1: RKNNLite.NPU_CORE_1,
                2: RKNNLite.NPU_CORE_2}
    def __init__(self, idx, model_path, in_q, out_q):
        super().__init__(daemon=True)
        self.in_q, self.out_q = in_q, out_q
        self.rknn = RKNNLite(verbose=False)
        assert self.rknn.load_rknn(model_path) == 0
        assert self.rknn.init_runtime(core_mask=self.CORE_MAP[idx]) == 0
        print(f'[Worker-{idx}] init OK')

    def run(self):
        while True:
            fid, frame = self.in_q.get()
            if fid is None:
                break
            h0, w0 = frame.shape[:2]
            img, r, (dw, dh) = letter_box(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = self.rknn.inference([np.expand_dims(img, 0)])

            branch, pair = 3, len(out) // 3
            boxes, cconfs, oconfs = [], [], []
            for i in range(branch):
                boxes.append(box_process(out[pair*i]))
                cconfs.append(out[pair*i+1])
                oconfs.append(np.ones_like(out[pair*i+1][:, :1, :, :], np.float32))
            merge = lambda xs: np.concatenate([x.transpose(0,2,3,1).reshape(-1, x.shape[1]) for x in xs])
            b, cls, s = filter_and_nms(merge(boxes), merge(cconfs), merge(oconfs))
            if b is not None:
                b = scale_boxes(b, (h0, w0), dw, dh, r)
            self.out_q.put((fid, b, cls, s))
        self.rknn.release()

# ================ 主程序 ================
def main():
    # 初始化ROS节点
    rospy.init_node("yolov8_detection_node")
    
    TARGET_FPS=30
    in_qs = [queue.Queue(6) for _ in range(3)]
    out_q = queue.Queue(12)
    workers = [InferenceWorker(i, MODEL_PATH, in_qs[i], out_q) for i in range(3)]
    for w in workers:
        w.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('无法打开摄像头')
        sys.exit()
        
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    #修改视频编码格式，否则默认只有5帧
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')          # Pythonic写法
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 640, 640)

    fid, t0 = 0, time.time()
    PERSIST_N = 7
    history = deque(maxlen=PERSIST_N)   # 保存最近N帧结果
    
    # 初始化参数
    drop_flag = False
    
    # 初始化ROS参数（避免删除时出错）
    param_names = ["center_x", "center_y", "cls_ids", "scores", "ready", "drop_flag"]
    for param_name in param_names:
        try:
            rospy.delete_param(param_name)
        except KeyError:
            pass  # 参数不存在，忽略
    
    # 设置初始值
    rospy.set_param("ready", drop_flag)
    rospy.set_param("drop_flag", drop_flag)
    rospy.set_param("center_x", 0.0)
    rospy.set_param("center_y", 0.0)
    rospy.set_param("cls_ids", [])
    rospy.set_param("scores", [])
    
    # FPS计算改进
    fps_history = deque(maxlen=30)

    try:
        while not rospy.is_shutdown():
            ok, frame = cap.read()
            if not ok:
                break

            # -------- 推理输入分发 --------
            target = fid % 3
            in_qs[target].put((fid, frame.copy()))
            fid += 1

            # -------- 取回推理结果 --------
            has_det = False
            while not out_q.empty():
                has_det = True
                _, boxes, cls_ids, scores = out_q.get()
                history.append((boxes, cls_ids, scores))

            # 若本帧没有新结果→沿用最近结果
            if not has_det and history:
                boxes, cls_ids, scores = history[-1]
            elif not history:        # 还没有任何推理输出
                boxes, cls_ids, scores = None, None, None

            # 绘制检测结果
            if boxes is not None:
                for box, cls, sc in zip(boxes, cls_ids, scores):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f'{get_class_name(cls)} {sc:.2f}',
                                (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,0,255), 2)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    cv2.circle(frame, (int(round(float(center_x))), int(round(float(center_y)))), 3, (0,0,255), -1)
                    
                    # 设置ROS参数
                    rospy.set_param("center_x", float(center_x))
                    rospy.set_param("center_y", float(center_y))
                    rospy.set_param("cls_ids", cls_ids.tolist())
                    rospy.set_param("scores", scores.tolist())
                    rospy.set_param("ready", True)
                    
                    # 更新drop_flag
                    drop_flag = True
                    rospy.set_param("drop_flag", drop_flag)
                    
                    print(f"检测到目标: 类别{cls}, 置信度{sc:.2f}, 中心({center_x:.1f}, {center_y:.1f})")
            else:
                # 没有检测到目标时，如果之前有检测，则重置drop_flag
                if drop_flag:
                    drop_flag = False
                    rospy.set_param("ready", False)
                    rospy.set_param("drop_flag", drop_flag)

            # 计算FPS（改进版）
            current_time = time.time()
            frame_time = current_time - t0
            current_fps = 1 / frame_time if frame_time > 0 else 0
            
            # 保存到历史记录
            fps_history.append(current_fps)
            
            # 计算平均FPS
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else current_fps
            
            # 更新时间戳
            t0 = current_time
            
            # 显示FPS（使用更稳定的平均FPS）
            cv2.putText(frame, f'FPS:{avg_fps:.1f}', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            # 显示画面
            if Flags:
                cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
                
            # 检查ROS是否关闭
            rospy.sleep(0.001)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for q in in_qs:
            q.put((None, None))
        for w in workers:
            w.join()

if __name__ == '__main__':
    main()
