#!/usr/bin/env python3
import rospy
import serial
import time

def main():
    rospy.init_node('param_monitor_minimal')
    
    # 串口初始化：尝试多个可能的端口，直到成功或超时
    port_list = ['/dev/ttyACM0', '/dev/ttyACM1']
    s = None
    for port in port_list:
        try:
            s = serial.Serial(port, 115200, timeout=1)
            rospy.loginfo(f"串口 {port} 连接成功")
            break
        except Exception as e:
            rospy.logwarn(f"连接串口 {port} 失败: {e}")
            continue
    if s is None:
        rospy.logfatal("无法打开任何串口，节点退出")
        return

    # 类别到指令的映射（可通过ROS参数动态配置，此处保留硬编码默认）
    CMD_MAP = {
        0: b'1',   # 类别0（例如 car）触发1号舵机
        5: b'2',   # 类别5（例如 tank）触发2号舵机
        6: b'3',   # 类别6（例如 red）触发3号舵机
    }
    DEFAULT_CMD = b'1'

    last_send_time = 0
    send_interval = 1.0  # 最小发送间隔（秒），避免重复触发
    rate = rospy.Rate(10)  # 10Hz 循环

    while not rospy.is_shutdown():
        try:
            # 从参数服务器读取标志，设置默认值防止参数缺失
            ready = rospy.get_param("ready", False)
            drop_flag = rospy.get_param("drop_flag", False)

            if ready and drop_flag:
                cls_ids = rospy.get_param("cls_ids", [])
                if cls_ids:  # 有检测到目标
                    now = time.time()
                    if now - last_send_time >= send_interval:
                        target_cls = cls_ids[0]  # 取第一个目标
                        cmd = CMD_MAP.get(target_cls, DEFAULT_CMD)
                        try:
                            s.write(cmd)
                            rospy.loginfo(f"发送指令 {cmd} (类别 {target_cls})")
                            last_send_time = now
                        except serial.SerialException as e:
                            rospy.logerr(f"串口写入失败: {e}")

            rate.sleep()

        except KeyboardInterrupt:
            break
        except Exception as e:
            rospy.logerr(f"未处理的错误: {e}")
            rate.sleep()

    if s is not None:
        s.close()
        rospy.loginfo("串口已关闭")

if __name__ == '__main__':
    main()
