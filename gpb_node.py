# gbp_node.py
import argparse, random
import torch, rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from std_msgs.msg import Bool
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from turtlesim.msg import Pose as TurtlesimPose

from robot_gbp_msgs.msg import GraphState, NodeState, FactorState, MotionCommand
from .gbp_algorithm import FactorGraph, GBPSettings, MeasModel
from .factors import (
    ReferenceModel, SquaredLoss,
    smooth_meas_fn_2d, smooth_jac_fn_2d,
    dR2RModel, RobotObstacleAvoidance
)
from .motion import allign_trajectory, euler_from_quaternion


class GBPNode(Node):
    obstacles = [torch.tensor([1.6, 0.5]), torch.tensor([3.5, 0.5]), torch.tensor([6.5, -0.3])]
    r = 0.3
    dt = 5.0
    timer_period = 1e-1

    def __init__(self, robot_id, nr_of_robots, r_, simulation_type, sigma_params):
        super().__init__('gbp_node_' + str(robot_id))

        # Params
        self.drop_rate = sigma_params[7]
        self.c_sigma_p = sigma_params[8]
        self.c_sigma_m = sigma_params[9]
        self.c_sigma_v = sigma_params[10]
        self.c_sigma_pull = sigma_params[11]
        self.c_sigma_obs = sigma_params[12]
        self.obstacle_scale = sigma_params[13]
        self.sep_scale = sigma_params[14]

        self.global_conv_n = 20
        self.energy_test = False
        self.pareto = self.energy_test

        self.sigma_p = sigma_params[0]
        self.sigma_m = sigma_params[1]
        self.sigma_ref = sigma_params[2]
        self.sigma_r2r = sigma_params[3]
        self.sigma_obstacle = sigma_params[4]

        self.prior_cov_diag     = torch.tensor([self.sigma_p, self.sigma_p, self.sigma_p/10])
        self.smoothing_cov_diag = torch.tensor([self.sigma_m, self.sigma_m])
        self.ref_cov_diag       = torch.tensor([self.sigma_ref, self.sigma_ref, self.sigma_ref])
        self.vicon_cov_diag     = torch.tensor([1e1, 1e1, 1e1])
        self.r2r_cov_diag       = torch.tensor([self.sigma_r2r])
        self.obstacle_cov_diag  = torch.tensor([self.sigma_obstacle])

        # Basic setup
        self.robot_id = robot_id
        self.nr_of_robots = nr_of_robots
        self.local_r = r_
        self.radii = [0.3, 0.3, 0.3, 0.3]
        if len(self.radii) < self.nr_of_robots:
            for _ in range(self.nr_of_robots - len(self.radii)):
                self.radii.append(self.local_r)

        self.tol = 1e-2
        self.adjacent_robots = [[i, (i + 1) % self.nr_of_robots] for i in range(self.nr_of_robots)]

        self.display_step_index = 0
        self.done = False
        self.display_delay_sec = 0.01
        self.display_timer = None

        self.n_states = 10
        self.current_step = 0
        self.initialized_first_step = False
        self.state = 'initial_position'
        self.robot_state = "not_initialized"
        self.reached_goal = False
        self.current_position = None
        self.local_convergence = False
        self.publishing_active = True
        self.published_speed = False
        self.last_five_beliefs = []
        self.rot_limit = 1.0
        self.speed_limit = 0.22
        self.max_predicted_speed = 0.0
        self.final_trajectory = [torch.tensor([0.0, 0.0, 0.0])] * self.n_states
        self.motion_status = "Initialized"

        # Pose source
        if simulation_type == 0:
            self.simulation_type = "pure"
            self.pose_type = TurtlesimPose
            self.pose_topic = f'B0{robot_id}/pose'
        elif simulation_type == 1:
            self.simulation_type = "turtle"
            self.pose_type = TurtlesimPose
            self.pose_topic = f'B0{robot_id}/pose'
        elif simulation_type == 2:
            self.simulation_type = "vicon"
            self.pose_type = TransformStamped
            self.pose_topic = f'vicon/B0{robot_id}/B0{robot_id}'
        else:
            self.get_logger().error("Invalid simulation type")

        # Graph + pubs/subs
        self.graph = FactorGraph(GBPSettings())

        self.publisher = self.create_publisher(GraphState, 'gbp_updates', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'robot_markers' + str(robot_id), 10)

        # latched obstacle markers
        latched = QoSProfile(depth=1)
        latched.reliability = QoSReliabilityPolicy.RELIABLE
        latched.durability  = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.obstacle_publisher = self.create_publisher(MarkerArray, 'obstacle_markers', latched)

        self.convergence_publisher = self.create_publisher(Bool, 'convergence_' + str(robot_id), 10)
        self.robot_executed_path_publisher = self.create_publisher(Path, 'robot_executed_path_' + str(robot_id), 10)
        self.path_publisher = self.create_publisher(Path, 'robot_path_' + str(robot_id), 10)
        self.motion_command_publisher = self.create_publisher(MotionCommand, f'/B0{robot_id}/mc', 10)
        self.mission_complete_publisher = self.create_publisher(Bool, f'/B0{robot_id}/mission_complete', 10)

        self.subscription = self.create_subscription(GraphState, 'gbp_updates', self.listener_callback, 10)
        self.position_subscriber = self.create_subscription(self.pose_type, self.pose_topic, self.position_callback, 10)
        self.global_convergence_subscriber = self.create_subscription(Bool, 'global_convergence', self.global_convergence_callback, 10)
        self.motion_command_subscriber = self.create_subscription(Bool, f'/B0{robot_id}/fdbk', self.motion_feedback_callback, 10)

        # Path msgs
        self.path = Path(); self.path.header.frame_id = 'map'
        self.robot_executed_path = Path(); self.robot_executed_path.header.frame_id = 'map'

        # Periodic publisher
        self.timer = self.create_timer(self.timer_period, self.interval_publisher)

        # Viz colors
        self.colors = [[i/self.nr_of_robots, 1.0 - i/self.nr_of_robots, 0.0, 1.0] for i in range(self.nr_of_robots)]

        self.iterations = 0
        self.peer_xy = [None] * self.nr_of_robots

        # Publish static obstacle markers once
        self._obstacle_markers = self._build_obstacle_markers()
        self.obstacle_publisher.publish(self._obstacle_markers)

        # Optional energy-test offset
        if self.energy_test:
            self.start_x = sigma_params[5]
            self.start_y = sigma_params[6]

        self.initialize_graph()

    def _build_obstacle_markers(self):
        arr = MarkerArray()
        mid = 0
        radius = float(self.obstacle_scale) * float(self.r)
        for o in self.obstacles:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.id = mid; mid += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(o[0])
            m.pose.position.y = float(o[1])
            m.pose.position.z = 0.3
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.5
            m.color.a = 1.0
            m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0
            m.lifetime = DurationMsg(sec=0, nanosec=0)
            arr.markers.append(m)
        return arr

    def initialize_graph(self):
        x_range = 10.0; y_range = 0.0; start_x = 0.0; start_y = 0.0
        if self.energy_test:
            start_x = self.start_x; start_y = self.start_y
        x_end = start_x + x_range; y_end = start_y + y_range

        xs = torch.linspace(start_x, x_end, self.n_states).float().unsqueeze(0).T
        ys = torch.linspace(start_y, y_end, self.n_states).float().unsqueeze(0).T
        zero = torch.zeros_like(xs)

        # Formation layout
        self.robot_positions = []
        if self.nr_of_robots > 4:
            for i in range(self.nr_of_robots):
                angle = 2 * 3.14159 * i / self.nr_of_robots
                r = self.local_r
                x = r * torch.cos(torch.tensor(angle))
                y = r * torch.sin(torch.tensor(angle))
                self.robot_positions.append(torch.tensor([x, y]))
        else:
            self.robot_positions = [
                torch.tensor([0.22, -self.radii[1]]),
                torch.tensor([0.22,  self.radii[0]]),
                torch.tensor([-0.22, self.radii[2]]),
                torch.tensor([-0.22,-self.radii[3]])
            ]

        # Reference path for this robot (straight)
        r_x = self.robot_positions[self.robot_id - 1][0].item()
        r_y = self.robot_positions[self.robot_id - 1][1].item()
        ref_path = torch.stack([xs + r_x, ys + r_y, zero], dim=1)
        while ref_path.dim() < 3:
            ref_path = ref_path.unsqueeze(-1)

        centroid_init_path = torch.stack([xs, ys, zero], dim=1)
        centroid_control_path = torch.stack([zero, zero, zero], dim=1)

        self.ref_path = allign_trajectory(ref_path, centroid_init_path)
        initial_path = self.ref_path + torch.randn_like(self.ref_path) * 0.00
        self.centroid_belief = torch.zeros_like(self.ref_path)
        self.centroid_control_belief = torch.zeros_like(centroid_control_path)
        self.initial_path = initial_path

        # Variables
        for i in range(self.n_states):
            self.graph.add_var_node(dofs=3, prior_mean=initial_path[i], prior_diag_cov=self.prior_cov_diag)

        # Smoothing
        for i in range(self.n_states - 1):
            self.graph.add_factor(
                [i, i + 1],
                torch.tensor([0., 0.]),
                MeasModel(smooth_meas_fn_2d, smooth_jac_fn_2d, SquaredLoss(2, self.smoothing_cov_diag)),
                factortype='smoothing'
            )

        # Anchor ref (start & end)
        for i in range(self.n_states):
            if i == 0 or i == self.n_states - 1:
                self.graph.add_factor(
                    [i],
                    torch.tensor([0., 0., 0.]),
                    ReferenceModel(self.ref_path[i], SquaredLoss(3, self.ref_cov_diag)),
                    factortype='reference'
                )

        # R2R separation
        adjacent_robots = self.adjacent_robots[self.robot_id - 1]
        dist_btwn_adj = torch.norm(self.robot_positions[adjacent_robots[1]] - self.robot_positions[adjacent_robots[0]])
        for i in range(self.n_states):
            self.graph.add_factor(
                [i],
                torch.tensor([0.]),
                dR2RModel(SquaredLoss(1, self.r2r_cov_diag), 2*dist_btwn_adj),
                factortype='R2R_' + str(self.robot_id)
            )

        # Obstacles
        for i in range(self.n_states):
            loss = SquaredLoss(1, self.obstacle_cov_diag)
            for obstacle in self.obstacles:
                self.graph.add_factor(
                    [i],
                    torch.tensor([0.]),
                    RobotObstacleAvoidance(obstacle, self.obstacle_scale * self.r, self.centroid_belief[i], loss),
                    factortype='obstacle'
                )

    def global_convergence_callback(self, msg: Bool):
        if msg.data:
            self.stop_publishing()
            if self.simulation_type == "pure":
                self.mission_complete_publisher.publish(Bool(data=True))
                self.display_results()
            else:
                self.start_path_execution()
        else:
            # stop robots
            current_pos = self.graph.var_nodes[self.current_step].belief.mean()
            motion_cmd = MotionCommand()
            motion_cmd.current_position = current_pos.squeeze().tolist()
            motion_cmd.target_position = current_pos.squeeze().tolist()
            motion_cmd.velocity = 0
            motion_cmd.duration = self.dt
            self.motion_command_publisher.publish(motion_cmd)
            self.resume_publishing()

    def diag_to_cov(self, diag):
        dofs = diag.shape[0]
        cov = torch.zeros(dofs, dofs, dtype=diag.dtype)
        cov[range(dofs), range(dofs)] = diag
        return cov

    def motion_feedback_callback(self, msg: Bool):
        if msg.data:
            self.motion_status = "Done"
            if self.current_step == self.n_states - 1:
                self.display_results()
                self.mission_complete_publisher.publish(Bool(data=True))
                self.stop_publishing()
            self.graph.var_nodes[self.current_step].belief.set_with_cov_form(
                self.current_position.unsqueeze(1), self.diag_to_cov(self.vicon_cov_diag)
            )
            self.resume_publishing()
            self.current_step += 1
            self.start_path_execution()
        else:
            self.motion_status = "Progress"

    def stop_publishing(self):
        if self.publishing_active:
            self.publishing_active = False

    def resume_publishing(self):
        self.local_convergence = False
        self.publishing_active = True
        self.interval_publisher()

    def serialize_graph_state(self):
        message = GraphState()
        message.robot_id = self.robot_id
        for node in self.graph.var_nodes:
            mean_list = node.belief.mean().squeeze().tolist()
            cov_list = node.belief.cov().diag().tolist()
            node_state = NodeState(id=node.variableID, mean=mean_list, covariance=cov_list)
            message.nodes.append(node_state)
        for factor in self.graph.factors:
            values_list = factor.measurement.flatten().tolist() if factor.measurement.dim() > 0 else [factor.measurement.item()]
            factor_state = FactorState(
                connected_node_ids=[n.variableID for n in factor.adj_var_nodes],
                values=values_list,
                type=factor.factor_type
            )
            message.factors.append(factor_state)
        return message

    def interval_publisher(self):
        if self.publishing_active:
            msg = self.serialize_graph_state()
            self.publisher.publish(msg)
        msg = Bool(); msg.data = self.local_convergence
        self.convergence_publisher.publish(msg)

    def listener_callback(self, msg):
        if msg.robot_id == self.robot_id:
            return
        nbrs = set(self.adjacent_robots[self.robot_id - 1])
        is_adjacent = (msg.robot_id in nbrs)
        is_centroid = (msg.robot_id == 999)
        if not (is_adjacent or is_centroid):
            return

        self.total_rx = getattr(self, "total_rx", 0) + 1
        if random.random() < self.drop_rate:
            self.dropped_messages = getattr(self, "dropped_messages", 0) + 1
            return

        if [msg.robot_id, self.robot_id] in self.adjacent_robots or [self.robot_id, msg.robot_id] in self.adjacent_robots:
            for i in range(min(len(msg.nodes), self.n_states)):
                for factor in self.graph.factors:
                    if "R2R" in factor.factor_type and i in factor.adj_vIDs:
                        received_measurement = torch.tensor(msg.nodes[i].mean[:2]).squeeze()
                        factor.meas_model.update_other_robot_pos(received_measurement)
                        factor.compute_factor()

        if is_centroid:
            for i in range(len(msg.nodes)):
                if i < self.n_states:
                    centroid_measurement = torch.tensor(msg.nodes[i].mean[:2]).squeeze()
                    self.centroid_belief[i] = torch.tensor(msg.nodes[i].mean).unsqueeze(1)
                    for factor in self.graph.factors:
                        if "obstacle" in factor.factor_type and i in factor.adj_vIDs:
                            factor.meas_model.update_centroid_belief(centroid_measurement)
                            factor.compute_factor()
                else:
                    self.centroid_control_belief[i - self.n_states] = torch.tensor(msg.nodes[i].mean).unsqueeze(1)

        self.iterations += 1
        self.start_time = self.get_clock().now()
        not_converged = self.graph.gbp_solve(
            n_iters=50,
            converged_threshold=1e-6,
            include_priors=True,
            animate=False
        )
        self.stop_time = self.get_clock().now()

        if not not_converged:
            poses = []
            for node in self.graph.var_nodes:
                pose = PoseStamped()
                pose.header.frame_id = 'world'
                pose.pose.position.x = node.belief.mean()[0].item()
                pose.pose.position.y = node.belief.mean()[1].item()
                poses.append(pose)
            self.path.poses = poses
            self.path_publisher.publish(self.path)

            if len(self.last_five_beliefs) < self.global_conv_n:
                self.last_five_beliefs.append(
                    torch.cat([var.belief.mean().unsqueeze(0) for var in self.graph.var_nodes], dim=0).detach()
                )
            self.check_convergence()

    def display_results_step(self):
        if self.display_step_index >= self.n_states:
            self.display_timer.cancel()
            self.display_timer = None
            self.robot_executed_path_publisher.publish(self.robot_executed_path)
            self.stop_publishing()
            self.done = True
            self.display_results()
            return

        current_pos = self.final_trajectory[self.display_step_index]
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.pose.position.x = current_pos[0].item()
        pose.pose.position.y = current_pos[1].item()
        self.robot_executed_path.poses.append(pose)
        self.robot_executed_path_publisher.publish(self.robot_executed_path)
        self.publish_robot_markers(up_to_step=self.display_step_index)
        self.display_step_index += 1

    def check_convergence(self):
        current_belief = torch.cat([var.belief.mean().unsqueeze(0) for var in self.graph.var_nodes], dim=0).detach()
        if len(self.last_five_beliefs) >= self.global_conv_n:
            self.last_five_beliefs.pop(0)
        self.last_five_beliefs.append(current_belief)

        average_belief = torch.mean(torch.stack(self.last_five_beliefs), dim=0)
        difference_norm = torch.norm(current_belief - average_belief)

        if difference_norm < self.tol:
            if not self.local_convergence:
                self.local_convergence = True
                msg = Bool(); msg.data = True
                self.convergence_publisher.publish(msg)
                self.stop_publishing()
        else:
            if self.local_convergence:
                self.local_convergence = False
                msg = Bool(); msg.data = False
                self.convergence_publisher.publish(msg)

    def display_results(self):
        if self.simulation_type == "pure" and not self.done:
            self.final_trajectory = [self.graph.var_nodes[i].belief.mean() for i in range(self.n_states)]
            self.peer_xy[self.robot_id - 1] = [(float(p[0]), float(p[1])) for p in self.final_trajectory]
            self.max_predicted_speed = max(abs(self.centroid_control_belief[i][0]) for i in range(self.n_states))
            self.display_step_index = 0
            self.robot_executed_path.poses.clear()
            self.display_timer = self.create_timer(self.display_delay_sec, self.display_results_step)
            return

        if self.current_step == self.n_states - 1 or self.done:
            self.robot_executed_path_publisher.publish(self.robot_executed_path)
            self.stop_publishing()

            # simple forward-sim for modeled_trajectory (kept, but not exported)
            modelled_trajectory = self.final_trajectory
            for i in range(1, len(modelled_trajectory)):
                prev = modelled_trajectory[i-1]
                curr = modelled_trajectory[i]
                v = self.centroid_control_belief[i-1][0]
                theta = torch.atan2(curr[1] - prev[1], curr[0] - prev[0])
                curr[0] = prev[0] + v * self.dt * torch.cos(theta)
                curr[1] = prev[1] + v * self.dt * torch.sin(theta)
                modelled_trajectory[i] = curr

    def position_callback(self, msg):
        if self.simulation_type == "turtle":
            x, y, theta = msg.x, msg.y, msg.theta
        elif self.simulation_type == "vicon":
            x, y = msg.transform.translation.x, msg.transform.translation.y
            quat = msg.transform.rotation
            _, _, theta = euler_from_quaternion(quat.x, quat.y, quat.z, quat.w)
        else:
            self.get_logger().error("Invalid simulation type")
            return

        self.current_position = torch.tensor([x, y, theta])

        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.pose.position.x = x
        pose.pose.position.y = y
        self.robot_executed_path.poses.append(pose)
        self.robot_executed_path_publisher.publish(self.robot_executed_path)

    def publish_robot_markers(self, up_to_step=None):
        marker_array = MarkerArray()
        id_counter = 0
        max_index = self.n_states if up_to_step is None else min(up_to_step + 1, self.n_states)

        for i in range(max_index):
            node = self.graph.var_nodes[i]
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = id_counter; id_counter += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = node.belief.mean()[0].item()
            marker.pose.position.y = node.belief.mean()[1].item()
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.1
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.colors[self.robot_id - 1]
            if i == up_to_step:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.2
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            marker.lifetime = DurationMsg(sec=0, nanosec=0)
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

        # obstacle markers (re-publish so late subscribers see them)
        obs_marker_array = MarkerArray()
        id_counter = 0
        for obstacle in self.obstacles:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = id_counter; id_counter += 1
            marker.type = Marker.CYLINDER
            marker.pose.position.x = obstacle[0].item()
            marker.pose.position.y = obstacle[1].item()
            marker.pose.position.z = 0.3
            marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.5
            marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
            marker.lifetime = DurationMsg(sec=0, nanosec=0)
            obs_marker_array.markers.append(marker)
        self.obstacle_publisher.publish(obs_marker_array)

    def start_path_execution(self):
        if self.motion_status == "Progress":
            return
        if self.current_step >= self.n_states - 1:
            self.display_results()
            return

        current_pos = self.graph.var_nodes[self.current_step].belief.mean()
        target_pos  = self.graph.var_nodes[self.current_step + 1].belief.mean()
        v = self.centroid_control_belief[self.current_step][0]

        if self.current_step == 0:
            current_pos = self.current_position
            target_pos  = self.graph.var_nodes[self.current_step].belief.mean()
            target_pos  = self.current_position
            v = torch.tensor([0.0])
            if current_pos is not None:
                dist = torch.norm(target_pos[:2] - current_pos[:2])
                v = 0 * dist

        if abs(v) > self.max_predicted_speed:
            self.max_predicted_speed = abs(v)
            v = torch.clamp(v, -self.speed_limit, self.speed_limit)

        if current_pos is None:
            return

        motion_cmd = MotionCommand()
        motion_cmd.current_position = current_pos.squeeze().tolist()
        motion_cmd.target_position  = target_pos.squeeze().tolist()
        motion_cmd.velocity = v.item()
        motion_cmd.duration = self.dt
        self.motion_command_publisher.publish(motion_cmd)
        self.final_trajectory[self.current_step] = current_pos


def main(args=None):
    parser = argparse.ArgumentParser(description='Start a GBP node with specified settings.')
    parser.add_argument('robot_id', type=int)
    parser.add_argument('nr_of_robots', type=int)
    parser.add_argument('r', type=float)
    parser.add_argument('simulation_type', type=int)
    parser.add_argument('sigma_p', type=float)
    parser.add_argument('sigma_m', type=float)
    parser.add_argument('sigma_ref', type=float)
    parser.add_argument('sigma_r2r', type=float)
    parser.add_argument('sigma_obstacle', type=float)
    parser.add_argument('start_x', type=float)
    parser.add_argument('start_y', type=float)
    parser.add_argument('drop_rate', type=float)
    parser.add_argument('c_sigma_p', type=float)
    parser.add_argument('c_sigma_m', type=float)
    parser.add_argument('c_sigma_v', type=float)
    parser.add_argument('c_sigma_pull', type=float)
    parser.add_argument('c_sigma_obs', type=float)
    parser.add_argument('obs_scale', type=float)
    parser.add_argument('sep_scale', type=float)

    args, unknown = parser.parse_known_args(args)
    rclpy.init(args=unknown)

    sigma_params = [
        args.sigma_p, args.sigma_m, args.sigma_ref, args.sigma_r2r, args.sigma_obstacle,
        args.start_x, args.start_y, args.drop_rate,
        args.c_sigma_p, args.c_sigma_m, args.c_sigma_v, args.c_sigma_pull, args.c_sigma_obs,
        args.obs_scale, args.sep_scale
    ]

    gbp_node = GBPNode(args.robot_id, args.nr_of_robots, args.r, args.simulation_type, sigma_params)
    rclpy.spin(gbp_node)


if __name__ == '__main__':
    main()
