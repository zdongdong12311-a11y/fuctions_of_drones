#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ============================================================================
# main_drop.py - 无人机任务指挥官
# 功能：读取航点文件，控制无人机飞往各个目标点，悬停等待视觉检测，
#       识别到目标后执行视觉伺服对准，下降并发送串口指令给Pico抛投。
# 依赖：MAVROS、Pixhawk/PX4飞控、Pico舵机控制器
# ============================================================================

import rospy
import serial
import time
import math
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from tf.transformations import euler_from_quaternion

# ================= 用户配置参数（需根据实际情况修改）=================
BAUD_RATE = 115200             # 串口波特率，必须与 Pico 代码一致

# 航点文件路径（格式：x y z 悬停时间(秒)）
ROUTE_FILE = "/home/nx/yuanshen-drone/src/realflight_modules/px4ctrl/route/test3.txt"

# 飞行高度设置
CRUISE_HEIGHT = 1.2    # 巡航高度（米），航点之间的飞行高度
DROP_HEIGHT = 0.6      # 抛投高度（米），下降到该高度后执行抛投
TANK_WAIT_HEIGHT = 1.0 # 目标点悬停检测高度（未使用，可忽略）

# 视觉伺服参数
IMG_WIDTH = 1920        # 相机图像宽度（像素）
IMG_HEIGHT = 1080       # 相机图像高度（像素）
CAM_CENTER_X = 960      # 图像中心 x 坐标（用于计算像素偏差）
CAM_CENTER_Y = 540      # 图像中心 y 坐标
PIXEL_THRESHOLD = 30    # 像素对准允许误差（小于此值认为已对准）
KP = 0.0025             # 比例控制系数：将像素误差转换为机体坐标系下的位置修正量（米/像素）
ALIGN_TIMEOUT = 8.0     # 视觉对准超时时间（秒），超时则强制抛投

# 目标类别映射（必须与 v888_drop.py 中的 CLASSES 对应）
# 普通目标：car(0), tank(5) ；特殊目标：red(6)
TARGET_CLASS_STD = [0, 5]      # 普通目标类别ID列表
TARGET_CLASS_RED = [6]         # 红色目标类别ID列表

# ================= 抛投位置偏移量配置（单位：米）=================
# 视觉对准后，由于投放装置可能不在飞机正中心，需对机身位置进行微调
OFFSET_STD_1_X = 0.0 ; OFFSET_STD_1_Y = 0.0   # 第一次普通目标投放的偏移量（机体坐标系：x前，y左）
OFFSET_STD_2_X = 0.0 ; OFFSET_STD_2_Y = 0.0   # 第二次普通目标投放的偏移量
OFFSET_RED_X   = 0.0 ; OFFSET_RED_Y   = 0.0   # 红色目标投放的偏移量

class MissionCommander:
    """任务指挥官类：负责无人机全程控制，包括起飞、航点飞行、视觉对准、抛投、降落。"""
    
    def __init__(self):
        """初始化节点、状态变量、串口、ROS通信等。"""
        rospy.init_node("mission_commander_node")

        # --- 状态变量 ---
        self.current_state = State()          # 飞控当前状态（如是否armed、模式）
        self.local_pose = PoseStamped()       # 无人机当前本地位置（来自mavros）
        self.current_yaw = 0.0                 # 当前偏航角（弧度）
        self.target_yaw = 0.0                  # 目标偏航角，保持与当前一致

        # 任务计数器
        self.std_drop_count = 0  # 普通目标已抛投次数 (0 -> 1 -> 2)
        self.red_dropped = False # 红色目标是否已抛投
        self.index_history = []  # 记录已处理过的目标ID，防止同一目标重复投放

        # --- 串口初始化（连接Pico）---
        # 尝试多个可能的串口端口，直到连接成功或全部失败
        port_list = ['/dev/ttyACM0', '/dev/ttyACM1']
        self.pico = None
        for port in port_list:
            try:
                self.pico = serial.Serial(port, BAUD_RATE, timeout=1)
                rospy.loginfo(f"Serial port {port} connected.")
                time.sleep(2)  # 等待Pico重启复位，确保串口稳定
                break
            except Exception as e:
                rospy.logwarn(f"连接串口 {port} 失败: {e}")
                continue
        if self.pico is None:
            rospy.logerr("FATAL: 无法打开任何串口，无人机仍可飞行但无法投放")

        # --- ROS 通信设置 ---
        self.sub_state = rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=10)
        self.sub_pose = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb, queue_size=10)
        self.pub_target = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)

        # 等待服务可用
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arm_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        # 定时器，20Hz 发布设定点（若不在IDLE状态）
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop) # 20Hz

        # 当前目标设定点 [x, y, z, yaw]（本地坐标系）
        self.setpoint = [0, 0, 0, 0]
        self.mode = "IDLE"  # 当前模式，控制定时器是否发布

        rospy.loginfo("Mission Commander Initialized.")

    # ================= 回调函数 =================
    def state_cb(self, msg):
        """飞控状态回调，更新 current_state。"""
        self.current_state = msg

    def pose_cb(self, msg):
        """本地位置回调，更新 local_pose 并计算当前偏航角。"""
        self.local_pose = msg
        orientation_list = [msg.pose.orientation.x, msg.pose.orientation.y,
                            msg.pose.orientation.z, msg.pose.orientation.w]
        # 将四元数转换为欧拉角，只取偏航角
        _, _, self.current_yaw = euler_from_quaternion(orientation_list)

    def control_loop(self, event):
        """定时器回调：若非IDLE模式，则发布当前设定点。"""
        if self.mode != "IDLE":
            self.publish_setpoint(*self.setpoint)

    # ================= 核心控制函数 =================
    def publish_setpoint(self, x, y, z, yaw):
        """
        发布位置设定点（FRAME_LOCAL_NED）到 /mavros/setpoint_raw/local。
        忽略速度和加速度，只使用位置和偏航角。
        """
        target = PositionTarget()
        target.header.stamp = rospy.Time.now()
        target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        # 掩码：忽略速度、加速度、偏航速率，只使用位置和偏航
        target.type_mask = PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ | \
                           PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | \
                           PositionTarget.IGNORE_YAW_RATE
        target.position.x = x
        target.position.y = y
        target.position.z = z
        target.yaw = yaw
        self.pub_target.publish(target)

    def arm_and_takeoff(self, height):
        """
        切换到OFFBOARD模式并解锁，然后起飞到指定高度。
        流程：先进入OFFBOARD（发送设定点），再解锁，最后飞往起始点上方。
        """
        rospy.loginfo("Wait for OFFBOARD...")
        # 循环等待直到进入OFFBOARD模式
        while not rospy.is_shutdown() and self.current_state.mode != "OFFBOARD":
            self.setpoint = [0, 0, height, self.current_yaw]  # 设定点：原地升空
            self.mode = "TAKEOFF"
            self.set_mode_client(custom_mode='OFFBOARD')
            rospy.sleep(0.2)

        rospy.loginfo("Arming...")
        # 循环等待直到解锁成功
        while not rospy.is_shutdown() and not self.current_state.armed:
            self.arm_client(True)
            rospy.sleep(0.2)

        rospy.loginfo(f"Taking off to {height}m...")
        self.target_yaw = self.current_yaw  # 保持当前偏航
        # 起飞：飞往当前起始点上方（高度height）
        start_x = self.local_pose.pose.position.x
        start_y = self.local_pose.pose.position.y
        self.goto_waypoint(start_x, start_y, height, timeout=15)

    def goto_waypoint(self, x, y, z, tolerance=0.15, timeout=15):
        """
        飞往指定本地坐标点 (x, y, z)，直到距离小于 tolerance 或超时。
        返回 True 表示成功到达，False 表示超时。
        """
        self.setpoint = [x, y, z, self.target_yaw]  # 更新目标设定点
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # 10Hz 检查距离
        while not rospy.is_shutdown():
            # 计算当前位置与目标点的距离
            dx = self.local_pose.pose.position.x - x
            dy = self.local_pose.pose.position.y - y
            dz = self.local_pose.pose.position.z - z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < tolerance:
                return True

            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn(f"Waypoint timeout! Dist: {dist:.2f}")
                return False
            rate.sleep()

    # ================= 视觉与坐标解算 =================
    def get_vision_info(self, target_type="std"):
        """
        从 ROS 参数服务器获取当前检测到的目标信息。
        优先使用新版本发布的列表参数（center_x_list, center_y_list），
        从中筛选出属于 target_type 类别的目标中得分最高的一个，返回其类别ID和中心像素坐标。
        若列表不存在或长度不匹配，则回退到旧版本的单点方式（最高得分目标的中心点）。
        
        参数：
            target_type: "std" 或 "red"，指定要找的目标类型
        返回：
            (detected_id, center_x, center_y) 或 None
        """
        try:
            ready = rospy.get_param('/ready', False)
            if not ready:
                return None

            cls_ids = rospy.get_param('/cls_ids', [])
            scores = rospy.get_param('/scores', [])

            if len(cls_ids) == 0:
                return None

            # 优先使用列表形式（v888_drop.py 修改后的新版本）
            if rospy.has_param('/center_x_list') and rospy.has_param('/center_y_list'):
                center_x_list = rospy.get_param('/center_x_list', [])
                center_y_list = rospy.get_param('/center_y_list', [])
                if len(center_x_list) == len(cls_ids) and len(center_y_list) == len(cls_ids):
                    # 确定要找的类别集合
                    if target_type == "red":
                        target_classes = TARGET_CLASS_RED
                    else:
                        target_classes = TARGET_CLASS_STD

                    # 收集所有属于目标类别的索引
                    valid_indices = [i for i, cid in enumerate(cls_ids) if cid in target_classes]
                    if not valid_indices:
                        return None

                    # 在这些索引中找得分最高的
                    best_idx = max(valid_indices, key=lambda i: scores[i])
                    best_cid = cls_ids[best_idx]
                    best_cx = center_x_list[best_idx]
                    best_cy = center_y_list[best_idx]
                    return best_cid, best_cx, best_cy
                else:
                    rospy.logwarn("center_x/y_list length mismatch, fallback to single center.")

            # 若列表不存在或长度不匹配，回退到旧方式（仅发布最高得分目标的中心点）
            center_x = rospy.get_param('/center_x', CAM_CENTER_X)
            center_y = rospy.get_param('/center_y', CAM_CENTER_Y)

            detected_id = -1
            if target_type == "red":
                for cid in cls_ids:
                    if cid in TARGET_CLASS_RED:
                        detected_id = cid
                        break
            else:
                for cid in cls_ids:
                    if cid in TARGET_CLASS_STD:
                        detected_id = cid
                        break

            if detected_id == -1:
                return None

            return detected_id, center_x, center_y

        except Exception as e:
            rospy.logwarn(f"get_vision_info error: {e}")
            return None

    def calculate_body_error(self, px_x, px_y):
        """
        根据目标在图像中的像素坐标计算机体坐标系下的误差。
        图像坐标系：原点左上角，x向右，y向下。
        机体坐标系（FLU）：x向前，y向左，z向上。
        转换关系：
            - 目标在图像下方（y大） -> 飞机需要向后飞（body_x 负）
            - 目标在图像右方（x大） -> 飞机需要向左飞（body_y 正？注意：此处推导需谨慎）
        实际代码采用：err_body_x = -err_pixel_y * KP, err_body_y = -err_pixel_x * KP
        即像素x偏差对应机体y方向修正，像素y偏差对应机体x方向修正，并乘以比例系数KP。
        返回 (err_body_x, err_body_y, err_pixel_x, err_pixel_y)
        """
        err_pixel_x = px_x - CAM_CENTER_X   # 像素x偏差（右正）
        err_pixel_y = px_y - CAM_CENTER_Y   # 像素y偏差（下正）

        # 将像素偏差转换为机体坐标系的期望位移修正量
        err_body_x = -err_pixel_y * KP       # 图像向下 => 机体向后（x负）
        err_body_y = -err_pixel_x * KP       # 图像向右 => 机体向左（y负？注意：根据FLU，y向左为正，但此处取负，需结合飞行方向测试调整）

        return err_body_x, err_body_y, err_pixel_x, err_pixel_y

    def visual_servo_align(self, target_type="std"):
        """
        视觉伺服对准：根据视觉反馈不断调整无人机位置，使目标逐渐移至图像中心。
        循环直至满足像素误差阈值或超时。
        """
        rospy.loginfo(f"Starting Visual Alignment for {target_type}...")
        start_time = time.time()
        rate = rospy.Rate(10)  # 10Hz控制频率

        while not rospy.is_shutdown():
            # 超时检查
            if time.time() - start_time > ALIGN_TIMEOUT:
                rospy.logwarn("Alignment Timeout! Proceeding to drop.")
                break

            # 获取当前视觉信息
            vision_data = self.get_vision_info(target_type)
            if vision_data is None:
                rate.sleep()
                continue

            det_id, cx, cy = vision_data

            # 计算机体坐标系误差
            err_b_x, err_b_y, px_err_x, px_err_y = self.calculate_body_error(cx, cy)

            # 判断是否已对准（像素误差小于阈值）
            if abs(px_err_x) < PIXEL_THRESHOLD and abs(px_err_y) < PIXEL_THRESHOLD:
                rospy.loginfo(f"Target Aligned! Error: {px_err_x:.1f}, {px_err_y:.1f}")
                break

            # 坐标变换：将机体坐标系下的修正量转换到世界坐标系（本地NED）
            cos_yaw = math.cos(self.current_yaw)
            sin_yaw = math.sin(self.current_yaw)

            # 世界坐标系修正量 = R_body_to_world * [err_b_x, err_b_y]^T
            world_err_x = err_b_x * cos_yaw - err_b_y * sin_yaw
            world_err_y = err_b_x * sin_yaw + err_b_y * cos_yaw

            # 更新目标设定点：当前位置 + 修正量
            new_target_x = self.local_pose.pose.position.x + world_err_x
            new_target_y = self.local_pose.pose.position.y + world_err_y

            self.setpoint = [new_target_x, new_target_y, self.local_pose.pose.position.z, self.current_yaw]
            rate.sleep()

    # ================= 动作执行 =================
    def perform_drop(self, servo_cmd, off_x, off_y, target_type="std"):
        """
        执行完整的抛投流程：
        1. 视觉伺服对准目标
        2. 应用投放偏移量（可选）
        3. 下降到抛投高度
        4. 发送串口指令给Pico舵机
        5. 上升回巡航高度
        """
        # 1. 视觉对准
        self.visual_servo_align(target_type)

        # 2. 应用偏移量（若不为0）
        if off_x != 0 or off_y != 0:
            rospy.loginfo(f"Applying Offset: x={off_x}, y={off_y}")
            cos_yaw = math.cos(self.current_yaw)
            sin_yaw = math.sin(self.current_yaw)
            # 将机体坐标系偏移量转换到世界坐标系
            w_off_x = off_x * cos_yaw - off_y * sin_yaw
            w_off_y = off_x * sin_yaw + off_y * cos_yaw

            t_x = self.local_pose.pose.position.x + w_off_x
            t_y = self.local_pose.pose.position.y + w_off_y
            self.goto_waypoint(t_x, t_y, self.local_pose.pose.position.z, tolerance=0.1)
            time.sleep(0.5)  # 短暂稳定

        # 3. 下降到抛投高度
        rospy.loginfo(f"Descending to {DROP_HEIGHT}m...")
        cx = self.local_pose.pose.position.x
        cy = self.local_pose.pose.position.y
        self.goto_waypoint(cx, cy, DROP_HEIGHT, tolerance=0.15)
        time.sleep(1.0)  # 等待机体稳定

        # 4. 发送抛投指令（串口）
        if self.pico and self.pico.is_open:
            try:
                self.pico.write(servo_cmd)  # 发送字节指令，如 b'1'
                rospy.loginfo(f"!!! DROP COMMAND SENT: {servo_cmd} !!!")
            except Exception as e:
                rospy.logerr(f"Serial Write Error: {e}")
        else:
            rospy.logwarn(f"Simulated Drop (No Serial): {servo_cmd}")

        time.sleep(1.0)  # 等待舵机动作完成

        # 5. 上升回巡航高度
        rospy.loginfo("Ascending...")
        self.goto_waypoint(cx, cy, CRUISE_HEIGHT)

    def run(self):
        """主任务流程：读取航点、起飞、依次访问航点、悬停检测、抛投、最终降落。"""
        # 读取航点文件
        try:
            with open(ROUTE_FILE, 'r') as f:
                route_lines = f.readlines()
        except Exception as e:
            rospy.logerr(f"Cannot read route file at {ROUTE_FILE}: {e}")
            return

        # 起飞
        self.arm_and_takeoff(CRUISE_HEIGHT)
        time.sleep(2)

        # 遍历航点
        for i, line in enumerate(route_lines):
            parts = line.split()
            if len(parts) < 4: continue
            wx, wy, wz, wait_time = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

            rospy.loginfo(f"==> Flying to Point {i+1}: ({wx}, {wy})")
            self.goto_waypoint(wx, wy, wz)

            # 悬停等待识别（最长 wait_time 秒）
            start_wait = time.time()
            while (time.time() - start_wait < wait_time):
                detected = False

                # --- 策略：优先检测红色目标（假设红色目标只在航线后半段出现，例如点2之后）---
                if (not self.red_dropped) and (i >= 2):
                    vision_red = self.get_vision_info("red")
                    if vision_red:
                        vid = vision_red[0]
                        if vid in TARGET_CLASS_RED:
                            rospy.loginfo("RED TARGET FOUND!")
                            self.perform_drop(b'3', OFFSET_RED_X, OFFSET_RED_Y, "red")
                            self.red_dropped = True
                            detected = True

                # --- 策略：检测普通目标（如果还没投完两次）---
                if (not detected) and (self.std_drop_count < 2):
                    vision_std = self.get_vision_info("std")
                    if vision_std:
                        vid = vision_std[0]
                        if (vid in TARGET_CLASS_STD) and (vid not in self.index_history):
                            rospy.loginfo(f"Standard Target {vid} Found!")

                            if self.std_drop_count == 0:
                                self.perform_drop(b'1', OFFSET_STD_1_X, OFFSET_STD_1_Y, "std")
                            else:
                                self.perform_drop(b'2', OFFSET_STD_2_X, OFFSET_STD_2_Y, "std")

                            self.std_drop_count += 1
                            self.index_history.append(vid)  # 记录已投放的目标ID
                            detected = True

                if detected:
                    # 如果在该点完成一次抛投，则跳出等待循环，前往下一个航点
                    break

                time.sleep(0.1)  # 短暂休眠，避免CPU过载

        rospy.loginfo("Mission Complete. Landing...")
        self.set_mode_client(custom_mode='AUTO.LAND')  # 切换至自动降落模式

if __name__ == "__main__":
    commander = MissionCommander()
    try:
        commander.run()
    except rospy.ROSInterruptException:
        pass

