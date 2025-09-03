import argparse, random, time, math
import numpy as np
import torch
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point

from robot_gbp_msgs.msg import GraphState, NodeState, FactorState
from .gbp_algorithm import FactorGraph, GBPSettings
from .factors import SquaredLoss, TernaryMotionFactor, CentroidPullModelDistributed
from .gbp_node import RobotObstacleAvoidance, GBPNode  # reuse obstacles/r/dt

def _mk_pose(x, y):
    p = PoseStamped()
    p.header.frame_id = 'world'
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    return p

class CentroidNode(Node):
    def __init__(self, nr_of_robots, sigma_params):
        super().__init__('centroid_node')

        # --- params / shared constants ---
        self.nr_of_robots = nr_of_robots
        self.n_states = 10
        self.robot_id = 999
        self.r, self.dt = GBPNode.r, GBPNode.dt
        self.obstacles = GBPNode.obstacles

        self.sigma_p, self.sigma_m, self.sigma_v, self.sigma_pull, self.sigma_obs = (
            sigma_params[0], sigma_params[1], sigma_params[2], sigma_params[3], sigma_params[4]
        )
        self.drop_rate = sigma_params[7]

        # --- graph + noises ---
        self.graph = FactorGraph(GBPSettings())
        self.prior_cov_diag         = torch.tensor([self.sigma_p, self.sigma_p, self.sigma_p])
        self.prior_control_cov_diag = torch.tensor([self.sigma_v, self.sigma_v, self.sigma_v])
        self.smoothing_cov          = torch.tensor([self.sigma_m])
        self.obstacle_cov_diag      = torch.tensor([self.sigma_obs])
        self.pull_in_cov            = torch.tensor([self.sigma_pull, self.sigma_pull])

        # --- pubs/subs ---
        self.publisher        = self.create_publisher(GraphState, 'gbp_updates', 10)
        self.path_pub         = self.create_publisher(Path, 'centroid_path', 10)
        self.markers_pub      = self.create_publisher(MarkerArray, 'centroid_markers', 10)
        self.start_end_pub    = self.create_publisher(MarkerArray, 'start_end_markers', 10)
        self.global_conv_pub  = self.create_publisher(Bool, 'global_convergence', 10)
        self.create_service(Empty, 'save_centroid_path', self._save_srv)

        self.create_subscription(GraphState, 'gbp_updates', self._on_msg, 10)
        self.conv_reports = [False] * self.nr_of_robots
        for i in range(1, nr_of_robots + 1):
            self.create_subscription(Bool, f'convergence_{i}', self._mk_conv_cb(i), 10)

        # path msg + loop
        self.path = Path(); self.path.header.frame_id = 'map'
        self.timer = self.create_timer(GBPNode.timer_period, self._tick)

        # global convergence state
        self.all_converged = False
        self.last_conv_time = None
        self.convergence_wait_s = 1.0
        self.publishing_active = True

        # simple straight-line reference & init
        self._build_graph()

    # ---------------- init helpers ----------------
    def _build_graph(self):
        x0, y0, x1, y1 = 0.0, 0.0, 10.0, 0.0
        xs = torch.linspace(x0, x1, self.n_states).float().unsqueeze(0).T
        ys = torch.linspace(y0, y1, self.n_states).float().unsqueeze(0).T
        v  = torch.zeros_like(xs); w = torch.zeros_like(xs)
        zero = torch.zeros_like(xs)

        self.centroid_ref_path    = torch.stack([xs, ys, zero], dim=1)
        v_avg = math.dist([x0, y0], [x1, y1]) / (self.n_states * self.dt)
        v[:] = v_avg
        self.centroid_control_path = torch.stack([v, w, zero], dim=1)

        # var nodes: states then controls
        for i in range(self.n_states):
            self.graph.add_var_node(3, self.centroid_ref_path[i], self.prior_cov_diag)
        for i in range(self.n_states - 1):
            self.graph.add_var_node(3, self.centroid_control_path[i], self.prior_control_cov_diag)

        # ternary motion
        for i in range(self.n_states - 1):
            self.graph.add_factor([i, i + 1, i + self.n_states],
                                  torch.tensor([0.]),
                                  TernaryMotionFactor(SquaredLoss(1, self.smoothing_cov), self.dt),
                                  factortype='ternary_motion')

        # obstacles
        self.centroid_belief = self.centroid_ref_path.clone()
        for i in range(self.n_states):
            loss = SquaredLoss(1, self.obstacle_cov_diag)
            for o in self.obstacles:
                self.graph.add_factor([i], torch.tensor([0.]),
                                      RobotObstacleAvoidance(o, 4*self.r, self.centroid_belief[i], loss),
                                      factortype='obstacle')

        # robot pull-in
        for r in range(1, self.nr_of_robots + 1):
            for i in range(self.n_states):
                self.graph.add_factor([i], torch.tensor([0., 0.]),
                                      CentroidPullModelDistributed(SquaredLoss(2, self.pull_in_cov)),
                                      factortype=f'centroid_pull_{r}')

    # ---------------- callbacks ----------------
    def _mk_conv_cb(self, rid):
        def cb(msg: Bool):
            self.conv_reports[rid - 1] = msg.data
            if all(self.conv_reports) and not self.all_converged:
                self.all_converged = True
                self.last_conv_time = self.get_clock().now()
            elif not all(self.conv_reports) and self.all_converged:
                self.all_converged = False
                self.publishing_active = True
                self.last_conv_time = None
        return cb

    def _on_msg(self, msg: GraphState):
        # robots pull the centroid
        if msg.robot_id != 999:
            if random.random() < self.drop_rate:
                return
            for i in range(min(len(msg.nodes), self.n_states)):
                for f in self.graph.factors:
                    if f'centroid_pull_{msg.robot_id}' == f.factor_type and i in f.adj_vIDs:
                        f.measurement = torch.tensor(msg.nodes[i].mean[:2]).squeeze()
                        f.compute_factor()

        # obstacle factors use current centroid belief
        for f in self.graph.factors:
            if f.factor_type == 'obstacle':
                for i in range(self.n_states):
                    if i in f.adj_vIDs:
                        cxy = self.graph.var_nodes[i].belief.mean()[0:2].detach().clone().squeeze()
                        f.meas_model.update_centroid_belief(cxy)
                        f.compute_factor()

        # solve
        self.start_time = time.time()
        _ = self.graph.gbp_solve(n_iters=30, converged_threshold=1e-6, include_priors=True, animate=False)
        self.stop_time = time.time()

        # publish centroid path & markers
        poses = []
        for i, node in enumerate(self.graph.var_nodes[:self.n_states]):
            x = node.belief.mean()[0].item() if i < self.n_states - 1 else self.centroid_ref_path[-1][0].item()
            y = node.belief.mean()[1].item() if i < self.n_states - 1 else self.centroid_ref_path[-1][1].item()
            poses.append(_mk_pose(x, y))
        self.path.poses = poses
        self._publish_markers()
        self.path_pub.publish(self.path)

    def _tick(self):
        # finalize global convergence after a small hold period
        if self.all_converged and self.last_conv_time:
            elapsed = (self.get_clock().now() - self.last_conv_time).nanoseconds / 1e9
            if elapsed > self.convergence_wait_s and self.publishing_active:
                self.publishing_active = False
                self.global_conv_pub.publish(Bool(data=True))

        if self.publishing_active:
            self.publisher.publish(self._serialize())

    # ---------------- util ----------------
    def _save_srv(self, _req, _res):
        # minimal: just acknowledge; robot nodes expect this service to exist
        self.get_logger().info("save_centroid_path request received (no-op).")
        return Empty.Response()

    def _serialize(self):
        msg = GraphState()
        msg.robot_id = self.robot_id
        for v in self.graph.var_nodes:
            m = v.belief.mean().squeeze().tolist()
            c = v.belief.cov().diag().tolist()
            msg.nodes.append(NodeState(id=v.variableID, mean=m, covariance=c))
        for f in self.graph.factors:
            vals = f.measurement.flatten().tolist() if f.measurement.dim() > 0 else [f.measurement.item()]
            msg.factors.append(FactorState(
                connected_node_ids=[n.variableID for n in f.adj_var_nodes],
                values=vals, type=f.factor_type
            ))
        return msg

    def _mk_marker(self, xy, mid, color=(0.0, 1.0, 0.0)):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = mid; m.type = Marker.CUBE; m.action = Marker.ADD
        m.pose.position.x = xy[0].item(); m.pose.position.y = xy[1].item(); m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.1
        m.color.a = 1.0; m.color.r, m.color.g, m.color.b = color
        m.lifetime = Duration(sec=0, nanosec=0)
        return m

    def _mk_x(self, xy, mid, color=(1.0, 1.0, 1.0), scale=0.3):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = mid; m.type = Marker.LINE_LIST; m.action = Marker.ADD
        m.scale.x = 0.05; m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = color
        off = scale / 2.0
        x, y = xy[0].item(), xy[1].item()
        m.points.extend([
            Point(x=x-off, y=y-off, z=0.1), Point(x=x+off, y=y+off, z=0.1),
            Point(x=x-off, y=y+off, z=0.1), Point(x=x+off, y=y-off, z=0.1)
        ])
        return m

    def _publish_markers(self):
        arr = MarkerArray()
        mid = 0
        for i in range(self.n_states):
            xy = self.graph.var_nodes[i].belief.mean()[:2]
            arr.markers.append(self._mk_marker(xy, mid)); mid += 1
        self.markers_pub.publish(arr)

        start_xy = self.centroid_ref_path[0][:2]
        end_xy   = self.centroid_ref_path[-1][:2]
        xs = MarkerArray()
        xs.markers.append(self._mk_x(start_xy, 9000, (0.0, 1.0, 0.0)))
        xs.markers.append(self._mk_x(end_xy,   9001, (1.0, 0.0, 0.0)))
        self.start_end_pub.publish(xs)

def main(args=None):
    parser = argparse.ArgumentParser(description='Start centroid node.')
    parser.add_argument('nr_of_robots', type=int)
    parser.add_argument('c_sigma_p', type=float)
    parser.add_argument('c_sigma_m', type=float)
    parser.add_argument('c_sigma_v', type=float)
    parser.add_argument('c_sigma_pull', type=float)
    parser.add_argument('c_sigma_obs', type=float)
    parser.add_argument('start_x', type=float)  # kept for compatibility, not used
    parser.add_argument('start_y', type=float)  # kept for compatibility, not used
    parser.add_argument('c_drop_rate', type=float)
    args, unknown = parser.parse_known_args(args)

    sigma_params = [
        args.c_sigma_p, args.c_sigma_m, args.c_sigma_v,
        args.c_sigma_pull, args.c_sigma_obs, args.start_x, args.start_y, args.c_drop_rate
    ]

    rclpy.init(args=unknown)
    node = CentroidNode(args.nr_of_robots, sigma_params)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
