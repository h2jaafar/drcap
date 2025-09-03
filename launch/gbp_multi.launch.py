from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Common sigma values (tune as needed)
    sigma_p      = "0.1"
    sigma_m      = "0.1"
    sigma_ref    = "0.1"
    sigma_r2r    = "0.1"
    sigma_obs    = "0.1"
    start_x      = "0.0"
    start_y      = "0.0"
    drop_rate    = "0.0"
    c_sigma_p    = "0.1"
    c_sigma_m    = "0.1"
    c_sigma_v    = "0.1"
    c_sigma_pull = "0.1"
    c_sigma_obs  = "0.1"
    c_drop_rate  = "0.0"

    nr_of_robots = "4"

    robots = []
    for rid in range(1, 5):
        robots.append(
            Node(
                package="drcap",
                executable="gbp_node",  # entry point for gbp_node.py
                name=f"gbp_node_{rid}",
                output="screen",
                arguments=[
                    str(rid), nr_of_robots, "0.3", "0",  # robot_id, nr_of_robots, r, simulation_type
                    sigma_p, sigma_m, sigma_ref, sigma_r2r, sigma_obs,
                    start_x, start_y,
                    drop_rate, c_sigma_p, c_sigma_m, c_sigma_v, c_sigma_pull,
                    c_sigma_obs, "1.0", "1.0", "0"  # obs_scale, sep_scale, centralized
                ],
            )
        )

    centroid = Node(
        package="drcap",
        executable="centroid_node",  # entry point for centroid node
        name="centroid_node",
        output="screen",
        arguments=[
            nr_of_robots,
            c_sigma_p, c_sigma_m, c_sigma_v, c_sigma_pull, c_sigma_obs,
            start_x, start_y, c_drop_rate,
        ],
    )

    return LaunchDescription(robots + [centroid])
