# Exercise 4: Don’t Crash! Tailing Behaviour

This repository contains implementation solutions for exercise 4. For information about the project, please read the report at:

[Nadeen Mohamed's site](https://sites.google.com/ualberta.ca/nadeen-cmput-412/written-reports/exercise-4) or [Celina Sheng's site](https://sites.google.com/ualberta.ca/csheng2-cmput-412/exercise-4)


## Structure

There are two packages in this file: duckiebot_detection and lane_follow. We will discuss the purpose of the python source files for each package (which are located inside the packages `src` folder).

### Duckiebot Detection

- `duckiebot_detection_node.py`: Implements a node that uses computer vision to detect a robot's circle pattern (located at the back of a Duckiebot). It publishes the circle pattern information to a rostopic.

- `duckiebot_distance_node.py`: Implements a node that uses the DuckiebotDetectionNode's detection information. It uses the detection information to calculate the distance between the source robot and the leader/detected robot. It also calculates the rotation of the leader robot. It publishes the distance and rotation to two separate rostopics.

### Lane Follow

- `duckiebot_follow_node.py`: Implements a node that tails behind a robot. It uses the distance and rotation information from the DuckiebotDetectionNode. It communicates to a rosservice in the LaneFollowNode if it no longer detects a leader robot to tail behind. In such case, it toggles autonomous lane following on.

- `lane_follow_node.py`: Implements a node to autonomously drive in a Duckietown lane. It contains a rosservice, which tells the node whether we want to lane follow or not.


## Execution:

To run the program, ensure that the variable `$BOT` stores your robot's host name, and run the following commands:

```
dts devel build -f -H $BOT
dts devel run -H $BOT
```

To shutdown the program, enter `CTRL + C` in your terminal.

## Credit:

This code is built from the 412 exercise 4 template that provides a boilerplate repository for developing ROS-based software in Duckietown (https://github.com/XZPshaw/CMPUT412503_exercise4).

Build on top of by Nadeen Mohamed and Celina Sheng.

Autonomous lane following code was also borrowed from Justin Francis.
