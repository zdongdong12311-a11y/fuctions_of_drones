#!/usr/bin/env python3
import rospy
import time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

def run_test():
    rospy.init_node('test_driver_node')
    
    # 1. 模拟 MAVROS 位置发布器
    fake_pose_pub = rospy.Publisher('/mavros/local_position/pose', PoseStamped, queue_size=10)
    
    print("=== 测试开始 ===")
    
    # 设置初始参数 (未准备好)
    rospy.set_param("ready", False)
    rospy.set_param("drop_flag", False)
    rospy.set_param("cls_ids", [])
    
    # --- 阶段 0: 模拟无人机起飞并悬停 ---
    print("\n[阶段 0] 模拟无人机起飞并悬停...")
    uav_pose = PoseStamped()
    uav_pose.header = Header(stamp=rospy.Time.now(), frame_id="map")
    uav_pose.pose.position.x = 0.0
    uav_pose.pose.position.y = 0.0
    uav_pose.pose.position.z = 10.0
    uav_pose.pose.orientation.w = 1.0 # 朝向正北
    
    # 持续发布位置，持续3秒
    for _ in range(30):
        uav_pose.header.stamp = rospy.Time.now()
        fake_pose_pub.publish(uav_pose)
        time.sleep(0.1)
        
    # --- 阶段 1: 发现第一个目标 (类别0) ---
    print("\n[阶段 1] 视觉发现目标 1 (ID:0)")
    rospy.set_param("ready", True)
    
    # 假设目标在图像中心左前方 (假设 K=0.01)
    # 图像中心 (960, 540)
    # 设置目标在像素 (960, 440) -> dy = -100 (上/前) -> body_x = 1.0m
    rospy.set_param("center_x", 960) 
    rospy.set_param("center_y", 440) 
    rospy.set_param("cls_ids", [0])
    rospy.set_param("drop_flag", True)
    
    # 持续发布位置，模拟飞行8.5秒 (超过8秒阈值)
    print(">>> 等待 8 秒飞行时间...")
    for i in range(85):
        uav_pose.header.stamp = rospy.Time.now()
        fake_pose_pub.publish(uav_pose)
        if i % 10 == 0:
            print(f"   模拟飞行中... {i/10}s")
        time.sleep(0.1)
        
    # --- 阶段 2: 目标 1 抛投完成，视觉丢失目标 ---
    print("\n[阶段 2] 模拟视觉丢失目标 (重置状态)")
    # 这一步非常关键！视觉节点必须在抛投后将 drop_flag 置为 False 或清空 cls_ids
    rospy.set_param("drop_flag", False)
    rospy.set_param("cls_ids", [])
    
    # 模拟寻找下一个目标的间隔时间 (3秒)
    for _ in range(30):
        uav_pose.header.stamp = rospy.Time.now()
        fake_pose_pub.publish(uav_pose)
        time.sleep(0.1)

    # --- 阶段 3: 发现第二个目标 (类别1) ---
    print("\n[阶段 3] 视觉发现目标 2 (ID:1)")
    
    # 假设目标在图像右侧 (假设你已经修正了 body_y 的负号)
    # 设置目标在像素 (1060, 540) -> dx = 100 (右)
    rospy.set_param("center_x", 1060) 
    rospy.set_param("center_y", 540) 
    rospy.set_param("cls_ids", [1])
    rospy.set_param("drop_flag", True)
    
    # 再次等待飞行 8.5 秒
    print(">>> 等待 8 秒飞行时间...")
    for i in range(85):
        uav_pose.header.stamp = rospy.Time.now()
        fake_pose_pub.publish(uav_pose)
        if i % 10 == 0:
            print(f"   模拟飞行中... {i/10}s")
        time.sleep(0.1)

    print("\n=== 测试结束 ===")

if __name__ == '__main__':
    try:
        run_test()
    except rospy.ROSInterruptException:
        pass