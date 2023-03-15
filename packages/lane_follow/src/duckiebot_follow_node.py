#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Header, Float32
from duckietown_msgs.msg import Twist2DStamped, LEDPattern
from duckietown_msgs.srv import SetFSMState

DEBUG = True

class DuckiebotFollowNode(DTROS):
  def __init__(self, node_name):
    super(DuckiebotFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.veh = rospy.get_param("~veh")

    # Subscribers
    self.distance_sub = rospy.Subscriber(
      f"/{self.veh}/duckiebot_distance_node/distance",
      Float32,
      self.cb_distance,
      queue_size=1,
      buff_size="20MB"
    )
    self.rotation_sub = rospy.Subscriber(
      f"/{self.veh}/duckiebot_distance_node/offset",
      Float32,
      self.cb_rotation,
      queue_size=1,
      buff_size="20MB"
    )
    
    # Publishers
    self.vel_pub = rospy.Publisher(
      f"/{self.veh}/car_cmd_switch_node/cmd",
      Twist2DStamped,
      queue_size=1
    )

    # Pose detection variables
    self.stale_time = 1
    rospy.Timer(rospy.Duration(self.stale_time), self.stale_detection)
    self.last_distance_detected_time = None
    self.distance_from_robot = None
    self.last_rotation_detected_time = None
    self.rotation_of_robot = None

    self.velocity = 0.3
    self.twist = Twist2DStamped(v = 0, omega = 0)

    rospy.wait_for_service('lane_following_service')
    self.lane_follow = rospy.ServiceProxy('lane_following_service', SetFSMState)
    self.lane_following = False

    # Distance PID variables
    self.distance_proportional = None

    self.distance_P = 3 # offset velocity
    self.distance_D = -0.5
    self.last_distance_error = 0
    self.last_distance_time = rospy.get_time()

    # Angle PID variables
    self.angle_proportional = None

    self.angle_P = 0.005
    self.angle_D = -0.0004
    self.last_angle_error = 0
    self.last_angle_time = rospy.get_time()
    
    # Duckiebot-following variables
    self.following_distance = 0.2

    # Initialize LED color-changing
    self.pattern = LEDPattern()
    self.pattern.header = Header()
    self.signalled = False

    # Shutdown hook
    rospy.on_shutdown(self.hook)

    self.loginfo("Initialized")
  
  def cb_distance(self, msg):
    self.distance_from_robot = msg.data
    self.distance_proportional = self.distance_from_robot - self.following_distance
    self.last_distance_detected_time = rospy.get_time()

  def cb_rotation(self, msg):
    self.rotation_of_robot = msg.data
    self.angle_proportional = self.rotation_of_robot
    self.last_rotation_detected_time = rospy.get_time()

  def stale_detection(self, _):
    """
    Remove Duckiebot detections if they are longer than the stale time
    """
    if (self.last_distance_detected_time and rospy.get_time() - self.last_distance_detected_time < self.stale_time) \
    or (self.last_rotation_detected_time and rospy.get_time() - self.last_rotation_detected_time < self.stale_time):
      self.distance_from_robot = None
      self.distance_proportional = None
      self.rotation_of_robot = None
      self.angle_proportional = None

      if not self.lane_following:
        self.lane_follow("True")
        self.lane_following = True
    elif self.lane_following:
        self.lane_follow("False")
        self.lane_following = False

  def drive(self):
    if self.lane_following:
      return
    
    # Determine Omega - based on lane-following
    if not self.distance_from_robot or not self.rotation_of_robot:
      self.twist.v = 0
    else:
      # Velocity control
      # P Term
      # distance_P = self.distance_proportional * self.distance_P

      # # D Term
      # distance_d_error = (self.distance_proportional - self.last_distance_error) / (rospy.get_time() - self.last_distance_time)
      # self.last_distance_error = self.distance_proportional
      # self.last_distance_time = rospy.get_time()
      # distance_D = distance_d_error * self.distance_D

      # self.twist.v = max(0, min(distance_P + distance_D, 0.5))
      if self.distance_from_robot > self.following_distance:
        self.twist.v = self.velocity
      else:
        self.twist.v = 0

      # Angle control
      # P Term
      angle_P = -self.angle_proportional * self.angle_P

      # D Term
      angle_d_error = (self.angle_proportional - self.last_angle_error) / (rospy.get_time() - self.last_angle_time)
      self.last_angle_error = self.angle_proportional
      self.last_angle_time = rospy.get_time()
      angle_D = angle_d_error * self.angle_D

      self.twist.omega = angle_P + angle_D

      print('angle:', self.angle_proportional)

      # Publish command
      if DEBUG:
        print('[DEBUG]', self.distance_proportional, self.twist.omega, self.twist.v)
    self.vel_pub.publish(self.twist)

  def hook(self):
    print("SHUTTING DOWN")
    self.twist.v = 0
    self.twist.omega = 0
    self.vel_pub.publish(self.twist)
    for i in range(8):
      self.vel_pub.publish(self.twist)

if __name__ == "__main__":
  node = DuckiebotFollowNode("duckiebot_follow_node")
  rate = rospy.Rate(8)  # 8hz
  while not rospy.is_shutdown():
    node.drive()
    rate.sleep()