#!/usr/bin/env python3

import rospy
import serial
import time
import math
from geometry_msgs.msg import PoseStamped, Point, PoseStamped
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class UAVTargetPublisher:
    def __init__(self):
        rospy.init_node('uav_target_publisher')
        
        # 串口初始化
        try:
            self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            print("串口连接成功")
        except Exception as e:
            print(f"串口连接失败: {e}")
            self.serial_port = None
        
        # 参数设置
        self.IMG_WIDTH = 1920
        self.IMG_HEIGHT = 1080
        self.K_VALUE = 0.01
        self.FLIGHT_TIME = 8.0  # 飞行时间（秒）
        
        # 状态变量
        self.target_sent = False
        self.drop_done = False
        self.position_send_time = None
        self.uav_pose = None
        self.takeoff_yaw = None  # 记录起飞时的偏航角
        
        # ROS话题发布者 - 改为发送PoseStamped
        self.target_pub = rospy.Publisher('/uav/target_pose', PoseStamped, queue_size=10)
        
        # 订阅无人机当前位置
        self.pose_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        
        print("无人机目标位置发布器启动...")
        print(f"飞行时间: {self.FLIGHT_TIME}秒")
    
    def pose_callback(self, msg):
        """无人机位置回调函数"""
        self.uav_pose = msg.pose
        
        # 如果是第一次收到姿态，记录起飞偏航角
        if self.takeoff_yaw is None:
            orientation_q = msg.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            _, _, yaw = euler_from_quaternion(orientation_list)
            self.takeoff_yaw = yaw
            print(f"记录起飞偏航角: {math.degrees(yaw):.1f}°")
    
    def image_to_world(self, center_x, center_y):
        """将图像坐标转换为世界坐标"""
        if self.uav_pose is None:
            return None, None
        
        # 图像中心
        img_center_x = self.IMG_WIDTH / 2
        img_center_y = self.IMG_HEIGHT / 2
        
        # 像素偏移
        dx_pixel = center_x - img_center_x
        dy_pixel = center_y - img_center_y
        
        # 无人机机体坐标系
        body_x = -dy_pixel * self.K_VALUE
        body_y = -dx_pixel * self.K_VALUE
        
        # 获取当前偏航角
        orientation_q = self.uav_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        
        # 旋转到世界坐标系
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        world_x = self.uav_pose.position.x + body_x * cos_yaw - body_y * sin_yaw
        world_y = self.uav_pose.position.y + body_x * sin_yaw + body_y * cos_yaw
        
        return world_x, world_y
    
    def send_target_pose(self, world_x, world_y, cls_id):
        """发送目标位置和姿态（始终朝向前方）"""
        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=rospy.Time.now(), frame_id="map")
        pose_msg.pose.position = Point(x=world_x, y=world_y, z=self.uav_pose.position.z)
        
        # 设置无人机朝向始终朝向前方（使用起飞时的偏航角）
        if self.takeoff_yaw is not None:
            # 使用起飞时的偏航角，保持无人机始终朝向前方
            # 如果需要，可以调整这里的偏航角设置
            quaternion = quaternion_from_euler(0, 0, self.takeoff_yaw)
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]
        else:
            # 如果没有记录起飞偏航角，使用当前偏航角
            pose_msg.pose.orientation = self.uav_pose.orientation
        
        self.target_pub.publish(pose_msg)
        self.target_sent = True
        self.drop_done = False
        self.position_send_time = time.time()
        
        print(f"发送目标位置: [{world_x:.3f}, {world_y:.3f}, {self.uav_pose.position.z:.3f}] m")
        print(f"目标类别: {cls_id}")
        print(f"等待飞行 {self.FLIGHT_TIME} 秒...")
        
        # 打印目标姿态信息
        orientation_list = [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, 
                           pose_msg.pose.orientation.z, pose_msg.pose.orientation.w]
        _, _, target_yaw = euler_from_quaternion(orientation_list)
        print(f"目标朝向: {math.degrees(target_yaw):.1f}°")
    
    def drop_cargo(self, cls_id):
        """抛投货物"""
        if self.serial_port is not None:
            if cls_id == 0:
                self.serial_port.write(bytes([1]))
                print("发送舵机命令: 1 (类别0)")
            elif cls_id == 1:
                self.serial_port.write(bytes([2]))
                print("发送舵机命令: 2 (类别1)")
            else:
                self.serial_port.write(bytes([1]))
                print("发送舵机命令: 1 (默认)")
        
        self.drop_done = True
        self.target_sent = False
        print("抛投完成，等待下一个目标...")
    
    def run(self):
        """主循环"""
        rate = rospy.Rate(10)  # 10Hz
        
        while not rospy.is_shutdown():
            try:
                # 检查无人机位置
                if self.uav_pose is None:
                    print("等待无人机位置...")
                    rate.sleep()
                    continue
                
                # 读取参数
                ready = rospy.get_param("ready", False)
                drop_flag = rospy.get_param("drop_flag", False)
                
                if not ready:
                    print("等待ready信号...")
                    rate.sleep()
                    continue
                
                center_x = rospy.get_param("center_x", 0)
                center_y = rospy.get_param("center_y", 0)
                cls_ids = rospy.get_param("cls_ids", [])
                
                # 检查是否有目标
                if not cls_ids or not drop_flag:
                    if self.target_sent:
                        print("目标丢失，重置状态...")
                        self.target_sent = False
                        self.drop_done = False
                        self.position_send_time = None
                    rate.sleep()
                    continue
                
                # 获取目标类别（使用最后一个）
                target_cls_id = cls_ids[-1]
                
                # 如果是新目标，发送位置和姿态
                if not self.target_sent:
                    world_x, world_y = self.image_to_world(center_x, center_y)
                    
                    if world_x is None or world_y is None:
                        print("坐标转换失败")
                        rate.sleep()
                        continue
                    
                    self.send_target_pose(world_x, world_y, target_cls_id)
                
                # 检查是否到抛投时间
                if self.target_sent and not self.drop_done:
                    elapsed_time = time.time() - self.position_send_time
                    
                    if elapsed_time >= self.FLIGHT_TIME:
                        print(f"飞行时间到 ({elapsed_time:.1f}秒)，开始抛投...")
                        self.drop_cargo(target_cls_id)
                    else:
                        # 显示剩余时间
                        remaining = self.FLIGHT_TIME - elapsed_time
                        print(f"剩余飞行时间: {remaining:.1f}秒")
                
                rate.sleep()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")
                rospy.sleep(1)
        
        # 关闭串口
        if self.serial_port is not None:
            self.serial_port.close()
            print("串口已关闭")

if __name__ == '__main__':
    try:
        publisher = UAVTargetPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass