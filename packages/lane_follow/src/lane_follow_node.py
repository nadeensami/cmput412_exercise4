#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from dt_apriltags import Detector
from turbojpeg import TurboJPEG, TJPF_GRAY
from image_geometry import PinholeCameraModel
import cv2
from std_msgs.msg import Header, ColorRGBA, Float32
from duckietown_msgs.msg import Twist2DStamped, LEDPattern

STOP_MASK = [(0, 75, 150), (5, 150, 255)]
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = False
ENGLISH = False
IS_FOLLOWING_ROBOT = False

class LaneFollowNode(DTROS):

  def __init__(self, node_name):
    super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.veh = rospy.get_param("~veh")

    # Publishers & Subscribers
    self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                   CompressedImage,
                   queue_size=1)
    self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                  CompressedImage,
                  self.callback,
                  queue_size=1,
                  buff_size="20MB")
    self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                     Twist2DStamped,
                     queue_size=1)
    
    self.distance_sub = rospy.Subscriber("/" + self.veh + "/duckiebot_distance_node/distance",
                  Float32,
                  self.cb_distance,
                  queue_size=1,
                  buff_size="20MB")
    
    self.distance_from_robot = None

    self.jpeg = TurboJPEG()

    self.loginfo("Initialized")

    # PID Variables
    self.proportional = None
    if ENGLISH:
      self.offset = -200
    else:
      self.offset = 200
    self.velocity = 0.4
    self.twist = Twist2DStamped(v=self.velocity, omega=0)

    self.P = 0.049
    self.D = -0.004
    self.last_error = 0
    self.last_time = rospy.get_time()

    # Stop variables
    self.stop = False
    self.last_stop_time = None
    self.stop_cooldown = 3
    self.stop_duration = 5
    self.stop_threshold_area = 5000 # minimun area of red to stop at
    self.stop_starttime = None
    
    # ====== April tag variables ======
    # Get static parameters    
    self.tag_size = 0.065
    self.rectify_alpha = 0.0

    # Initialize detector
    self.at_detector = Detector(
      searchpath = ['apriltags'],
      families = 'tag36h11',
      nthreads = 1,
      quad_decimate = 1.0,
      quad_sigma = 0.0,
      refine_edges = 1,
      decode_sharpening = 0.25,
      debug = 0
    )

    # Initialize static parameters from camera info message
    camera_info_msg = rospy.wait_for_message(f'/{self.veh}/camera_node/camera_info', CameraInfo)
    self.camera_model = PinholeCameraModel()
    self.camera_model.fromCameraInfo(camera_info_msg)
    H, W = camera_info_msg.height, camera_info_msg.width
    # find optimal rectified pinhole camera
    rect_K, _ = cv2.getOptimalNewCameraMatrix(
      self.camera_model.K, self.camera_model.D, (W, H), self.rectify_alpha
    )
    # store new camera parameters
    self._camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])

    self._mapx, self._mapy = cv2.initUndistortRectifyMap(
      self.camera_model.K, self.camera_model.D, None, rect_K, (W, H), cv2.CV_32FC1
    )

    self.intersection_apriltags =  [169, 162, 153, 133, 62, 58]
    self.inner_intersection_apriltags = [133, 169, 162, 58]

    self.last_detected_apriltag = None

    # Apriltag timer
    self.publish_hz = 1
    self.timer = rospy.Timer(rospy.Duration(1 / self.publish_hz), self.cb_apriltag_timer)
    self.last_message = None

    # Initialize LED color-changing
    self.color_publisher = rospy.Publisher(f'/{self.veh}/led_emitter_node/led_pattern', LEDPattern, queue_size = 1)
    self.pattern = LEDPattern()
    self.pattern.header = Header()
    self.signalled = False

    # Shutdown hook
    rospy.on_shutdown(self.hook)

  def callback(self, msg):
    self.last_message = msg

    img = self.jpeg.decode(msg.data)
    crop = img[300:-1, :, :]
    crop_width = crop.shape[1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Mask for road lines
    roadMask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
    # crop = cv2.bitwise_and(crop, crop, mask=roadMask)
    contours, _ = cv2.findContours(
      roadMask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE
    )

    # Search for lane in front
    max_area = 20
    max_idx = -1
    for i in range(len(contours)):
      area = cv2.contourArea(contours[i])
      if area > max_area:
        max_idx = i
        max_area = area

    if max_idx != -1:
      M = cv2.moments(contours[max_idx])
      try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.proportional = cx - int(crop_width / 2) + self.offset
        if DEBUG:
          cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
          cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
      except:
        pass
    else:
      self.proportional = None
    
    # See if we need to look for stop lines
    if self.stop or (self.last_stop_time and rospy.get_time() - self.last_stop_time < self.stop_cooldown):
      if DEBUG:
        rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
        self.pub.publish(rect_img_msg)
      return
    
    # Mask for stop lines
    stopMask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
    # crop = cv2.bitwise_and(crop, crop, mask=stopMask)
    stopContours, _ = cv2.findContours(
      stopMask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE
    )

    # Search for lane in front
    max_area = self.stop_threshold_area
    max_idx = -1
    for i in range(len(stopContours)):
      area = cv2.contourArea(stopContours[i])
      if area > max_area:
        max_idx = i
        max_area = area

    if max_idx != -1:
      M = cv2.moments(stopContours[max_idx])
      try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.stop = True
        self.stop_starttime = rospy.get_time()
        if DEBUG:
          cv2.drawContours(crop, stopContours, max_idx, (0, 255, 0), 3)
          cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
      except:
        pass
    else:
      self.stop = False

    if DEBUG:
      rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
      self.pub.publish(rect_img_msg)

  def cb_apriltag_timer(self, _):
    '''
    Callback for timer
    '''

    msg = self.last_message
    if not msg:
      return

    self.last_detected_apriltag = None
    # turn image message into grayscale image
    img = self.jpeg.decode(msg.data, pixel_format=TJPF_GRAY)
    # run input image through the rectification map
    img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)

    # detect tags
    tags = self.at_detector.detect(img, True, self._camera_parameters, self.tag_size)

    # Only save the april tag if it's within a close distance
    min_tag_distance = 2
    for tag in tags:

      distance = tag.pose_t[2][0]
      if distance > min_tag_distance:
        continue

      # save tag id if we're about to go to an intersection
      if tag.tag_id in self.intersection_apriltags:
        self.last_detected_apriltag = tag.tag_id
    
  def cb_distance(self, msg):
    self.distance_from_robot = msg.data

  def drive(self):
    if self.stop:
      if rospy.get_time() - self.stop_starttime < self.stop_duration:
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)

        if not IS_FOLLOWING_ROBOT and not self.signalled and self.last_detected_apriltag in self.inner_intersection_apriltags:
          if ENGLISH:
            self.change_color("left")
          else:
            self.change_color("right")
          self.signalled = True

      else:
        self.stop = False
        self.last_stop_time = rospy.get_time()
        self.change_color(None)
        self.signalled = False
    else:
      # Determine Velocity - based on if we're following a Duckiebot or not
      if self.distance_from_robot is None:
        self.twist.v = self.velocity
      else:
        # Use the PID here
        pass

      # Determine Omega - based on lane-following
      if self.proportional is None:
        self.twist.omega = 0
      else:
        # P Term
        P = -self.proportional * self.P

        # D Term
        d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
        self.last_error = self.proportional
        self.last_time = rospy.get_time()
        D = d_error * self.D

        self.twist.omega = P + D

      # Publish command
      if DEBUG:
        # self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)
        print(self.proportional, P, D, self.twist.omega, self.twist.v)
      self.vel_pub.publish(self.twist)
      

  def change_color(self, turn_signal):
    '''
    Code for this function was inspired by 
    "duckietown/dt-core", file "led_emitter_node.py"
    Link: https://github.com/duckietown/dt-core/blob/daffy/packages/led_emitter/src/led_emitter_node.py
    Author: GitHub user liampaull
    '''

    self.pattern.header.stamp = rospy.Time.now()
    rgba_yellow = ColorRGBA()
    rgba_none = ColorRGBA()

    rgba_yellow.r = 1.0
    rgba_yellow.g = 1.0
    rgba_yellow.b = 0.0
    rgba_yellow.a = 0.0

    rgba_none.r = 0.0
    rgba_none.g = 0.0
    rgba_none.b = 0.0
    rgba_none.a = 0.0

    # default: turn off
    self.pattern.rgb_vals = [rgba_none] * 5

    if turn_signal == "right":
      self.pattern.rgb_vals = [rgba_none, rgba_none, rgba_yellow, rgba_yellow, rgba_none]
    elif  turn_signal == "left":
      self.pattern.rgb_vals = [rgba_yellow, rgba_none, rgba_none, rgba_none, rgba_yellow]
      
    self.color_publisher.publish(self.pattern)

  def hook(self):
    print("SHUTTING DOWN")
    self.twist.v = 0
    self.twist.omega = 0
    self.vel_pub.publish(self.twist)
    for i in range(8):
      self.vel_pub.publish(self.twist)

if __name__ == "__main__":
  node = LaneFollowNode("lanefollow_node")
  rate = rospy.Rate(8)  # 8hz
  while not rospy.is_shutdown():
    node.drive()
    rate.sleep()