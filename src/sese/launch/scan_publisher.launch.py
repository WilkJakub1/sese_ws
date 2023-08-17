import os

from ament_index_python.packages import get_package_share_directory

from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription, actions, conditions
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch.actions import ExecuteProcess



def generate_launch_description():


    # Include the robot_state_publisher launch file, provided by our own package. Force sim time to be enabled
    # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!

    package_name='sese' #<--- CHANGE ME

    package_share_directory = get_package_share_directory(package_name)

    world_file = os.path.join(package_share_directory, "worlds", "empty.world")

    default_rviz_config_path = os.path.join(package_share_directory, 'rviz/urdf_config.rviz')

    # nav2 = IncludeLaunchDescription(
    #             PythonLaunchDescriptionSource([os.path.join(
    #                 get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')]),
    #          )

    scan_publisher = Node(package=package_name, executable='scan_publisher.py', name='scan_publisher', output='screen')

    return LaunchDescription([
        actions.DeclareLaunchArgument(name='use_sim_time', default_value='True',
                                            description='Flag to enable use_sim_time'),
        actions.DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
                                            description='Absolute path to rviz config file'),
        scan_publisher,
        # nav2, 

    ])