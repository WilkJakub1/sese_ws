How to run:
* install ros2 foxy and nav2 package (https://navigation.ros.org/getting_started/index.html)
* copy the sese_ws repository
* `cd sese_ws`
* `source install/setup.bash` to source the workspace
* `colcon build --symlink-install` to build the workspace
* `ros2 launch sese launch_sim.launch.py` to start simulation, rviz and launch 'online_async_launch.py' from slam_toolbox package 
* `ros2 launch sese nav2_launch.py` to launch 'navigation_launch.py' from nav2_bringup package and scan_publisher node which transforms camera detections to lidar /scan topic
* `cd ros2_bags/_2023-03-24-19-19-06`
* `ros2 bag play _2023-03-24-19-19-06.db3`

launch_sim.launch.py and nav2_launch.py should be one launch file but when I did that navigation_launch.py doesn't work properly. That's why they are separate launch files.That's sth to work out in the future.  
