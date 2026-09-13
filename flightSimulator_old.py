import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim

from ursina import *
import random
import math
from collections import deque

app = Ursina()

# ---------------------------------------------------------------------------
# GLOBAL GAME STATE
# ---------------------------------------------------------------------------
game_paused = False

# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------
Sky()

ground = Entity(
    model='plane',
    scale=10000,
    texture='grass',
    texture_scale=(1000, 1000),
    y=0
)

# ---------------------------------------------------------------------------
# PLAYER AIRCRAFT
# ---------------------------------------------------------------------------
airplane = Entity(position=(0, 10, 0))

body = Entity(
    parent=airplane,
    model='cube',
    color=color.azure,
    scale=(3, 1, 5)
)

wings = Entity(
    parent=airplane,
    model='cube',
    color=color.blue,
    scale=(12, 0.1, 1.2),
    position=(0, 0, 0)
)

speed = 50
turn_speed = 45
vertical_velocity = 0
gravity = 9.8

# Camera setup
camera.parent = airplane
camera.position = (0, 35, -130)
camera.rotation = (15, 0, 0)

# ---------------------------------------------------------------------------
# FLIGHT HUD
# ---------------------------------------------------------------------------
hud_text = Text(
    position=(-0.85, 0.45),
    scale=1.2,
    color=color.green
)

hud_controls_info = Text(
    text="[M] Menu / Pause | [P] Pause | [O] Save | [I] Load | [R] Reset",
    position=(0.12, 0.48),
    scale=0.8,
    color=color.light_gray
)

# Artificial Horizon UI
hud_horizon_pivot = Entity(parent=camera.ui, position=(0, 0))
horizon_line = Entity(
    parent=hud_horizon_pivot,
    model='quad',
    color=color.green,
    scale=(0.35, 0.003)
)

# Fixed Reticle
reticle_left = Entity(parent=camera.ui, model='quad', color=color.lime, scale=(0.06, 0.004), position=(-0.08, 0))
reticle_right = Entity(parent=camera.ui, model='quad', color=color.lime, scale=(0.06, 0.004), position=(0.08, 0))
reticle_center = Entity(parent=camera.ui, model='quad', color=color.lime, scale=(0.012, 0.012), position=(0, 0))


# ---------------------------------------------------------------------------
# WINGMAN AI (Neural Network)
# ---------------------------------------------------------------------------
OBS_DIM = 13
HIDDEN_DIM = 16
ACT_DIM = 3


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, x):
        h = self.trunk(x)
        mean = torch.tanh(self.mean_head(h))
        value = self.value_head(h)
        return mean, value


class WingmanAI:
    def __init__(self, leader, offset=Vec3(25, 0, 0), name="Wingman", body_color=None):
        self.leader = leader
        self.offset = offset
        self.name = name
        self.save_path = f"wingman_brain_{name.lower().replace(' ', '_')}.pt"

        body_color = body_color or color.lime
        self.entity = Entity(
            model='cube', color=body_color, scale=(3, 1, 5),
            position=leader.position + offset
        )
        Entity(parent=self.entity, model='cube', color=color.green, scale=(5, 0.1, 0.3))

        self.target_dot = Entity(
            model='sphere',
            color=color.red,
            scale=1.5,
            unlit=True
        )

        self.speed = 50.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0

        self.net = PolicyNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=3e-4)
        self.gamma = 0.95

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

        self.steps_per_update = 60
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0.0

        self.paused = False
        self.current_dist_to_target = 0.0

    def _formation_target(self):
        return (
            self.leader.position
            + self.leader.right * self.offset.x
            + self.leader.up * self.offset.y
            + self.leader.forward * self.offset.z
        )

    def get_state(self, all_wingmen):
        target = self._formation_target()
        rel = target - self.entity.position
        local_rel = Vec3(rel.dot(self.entity.right), rel.dot(self.entity.up), rel.dot(self.entity.forward))

        heading_diff = (self.leader.rotation_y - self.entity.rotation_y)
        heading_diff = (heading_diff + 180) % 360 - 180

        min_peer_dist = float('inf')
        closest_peer_rel = Vec3(0, 0, 0)

        for peer in all_wingmen:
            if peer is self:
                continue
            d = distance(self.entity.position, peer.entity.position)
            if d < min_peer_dist:
                min_peer_dist = d
                closest_peer_rel = peer.entity.position - self.entity.position

        local_peer_rel = Vec3(
            closest_peer_rel.dot(self.entity.right),
            closest_peer_rel.dot(self.entity.up),
            closest_peer_rel.dot(self.entity.forward)
        ) if min_peer_dist != float('inf') else Vec3(0, 0, 0)

        obs = [
            local_rel.x / 50.0, local_rel.y / 50.0, local_rel.z / 50.0,
            self.rotation_x / 90.0, self.rotation_z / 90.0,
            heading_diff / 180.0,
            (self.speed - 50.0) / 50.0,
            self.entity.y / 100.0,
            self.leader.position.x / 500.0, self.leader.position.z / 1500.0,
            local_peer_rel.x / 50.0, local_peer_rel.y / 50.0, local_peer_rel.z / 50.0
        ]
        return torch.tensor(obs, dtype=torch.float32), target, min_peer_dist

    def update(self, all_wingmen):
        if game_paused:
            return

        state, target, min_peer_dist = self.get_state(all_wingmen)
        self.target_dot.position = target

        mean, value = self.net(state)
        std = self.net.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = mean.detach() if self.paused else dist.sample()
        action_clamped = torch.clamp(action, -1, 1)
        pitch_cmd, roll_cmd, throttle_cmd = action_clamped.tolist()

        self.rotation_x += pitch_cmd * turn_speed * time.dt
        self.rotation_z += roll_cmd * turn_speed * time.dt
        self.rotation_y += self.rotation_z * 0.8 * time.dt
        self.speed = max(15.0, min(120.0, self.speed + throttle_cmd * 10 * time.dt))

        self.entity.rotation = (self.rotation_x, self.rotation_y, self.rotation_z)
        self.entity.position += self.entity.forward * self.speed * time.dt

        lift = (self.speed / 50.0) ** 2 * gravity
        vert_vel = lift - gravity
        self.entity.y = max(0.5, self.entity.y + vert_vel * time.dt)

        self.current_dist_to_target = distance(self.entity.position, target)
        reward = -self.current_dist_to_target / 20.0
        reward -= (abs(pitch_cmd) + abs(roll_cmd)) * 0.01

        self.episode_reward += reward

        if not self.paused:
            self.log_probs.append(dist.log_prob(action).sum())
            self.values.append(value.squeeze())
            self.rewards.append(reward)
            self.entropies.append(dist.entropy().sum())

        self.step_count += 1

        # Continuous learning weight update step
        if not self.paused and self.step_count >= self.steps_per_update:
            self._learn(bootstrap=True)

    def _respawn(self):
        self.entity.position = (
            self.leader.position
            + self.leader.right * self.offset.x
            + self.leader.up * self.offset.y
            + self.leader.forward * self.offset.z
        )
        self.rotation_x, self.rotation_y, self.rotation_z = (
            self.leader.rotation_x, self.leader.rotation_y, self.leader.rotation_z
        )
        self.entity.rotation = (self.rotation_x, self.rotation_y, self.rotation_z)
        self.speed = 50.0

    def hard_reset(self):
        self._respawn()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.step_count = 0
        self.episode_reward = 0.0

    def _learn(self, bootstrap=True):
        if not self.rewards:
            return

        with torch.no_grad():
            if bootstrap:
                state, _, _ = self.get_state([])
                _, next_value = self.net(state)
                R = next_value.squeeze()
            else:
                R = torch.tensor(0.0)

        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)

        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        advantages = returns.detach() - values
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_bonus = entropies.mean()

        loss = policy_loss + 0.5 * value_loss - 0.001 * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.episode_count += 1
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.step_count = 0
        self.episode_reward = 0.0

    def save(self):
        torch.save(self.net.state_dict(), self.save_path)

    def load(self):
        try:
            self.net.load_state_dict(torch.load(self.save_path))
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# WINGMEN CREATION
# ---------------------------------------------------------------------------
def generate_right_wingmen(n):
    bots = []
    right_side_offset = Vec3(25, 0, 0)

    for i in range(n):
        hue = (i * 360 / n)
        bot_color = color.hsv(hue, 0.8, 0.9)

        bots.append(WingmanAI(
            leader=airplane,
            offset=right_side_offset,
            name=f"Bot-{i + 1}",
            body_color=bot_color
        ))
    return bots


NUM_WINGMEN = 10
wingmen = generate_right_wingmen(NUM_WINGMEN)


# ---------------------------------------------------------------------------
# MAIN MENU & OVERLAYS
# ---------------------------------------------------------------------------
menu_container = Entity(parent=camera.ui, enabled=False, position=(0, 0, -0.05))

menu_bg = Entity(
    parent=menu_container,
    model='quad',
    color=color.rgba(12, 16, 24, 245),
    scale=(1.70, 0.95),
    position=(0, 0, 0)
)

menu_title = Text(
    text="TACTICAL MAIN MENU",
    parent=menu_container,
    position=(-0.80, 0.43, -0.01),
    scale=1.5,
    color=color.cyan
)

menu_status = Text(
    text="",
    parent=menu_container,
    position=(-0.80, 0.37, -0.01),
    scale=1.0,
    color=color.yellow
)

graph_container = Entity(parent=menu_container, position=(-0.42, -0.05, -0.01))
radar_container = Entity(parent=menu_container, position=(0.45, -0.05, -0.01))


# ---------------------------------------------------------------------------
# NEURAL NETWORK GRAPH VISUALIZER
# ---------------------------------------------------------------------------
def build_network_graph():
    for child in list(graph_container.children):
        destroy(child)

    Text(
        text="Policy Network Weights",
        parent=graph_container,
        position=(-0.35, 0.38, -0.02),
        scale=0.9,
        color=color.lime
    )

    net = wingmen[0].net

    w1 = net.trunk[0].weight.data
    w2 = net.trunk[2].weight.data
    w3 = net.mean_head.weight.data

    layers = [OBS_DIM, HIDDEN_DIM, HIDDEN_DIM, ACT_DIM]
    x_positions = [-0.32, -0.10, 0.10, 0.32]

    node_positions = []
    for l_idx, num_nodes in enumerate(layers):
        x = x_positions[l_idx]
        layer_nodes = []
        y_span = 0.55
        y_step = y_span / max(1, num_nodes - 1)
        y_start = y_span / 2.0

        for n_idx in range(num_nodes):
            y = y_start - n_idx * y_step if num_nodes > 1 else 0
            layer_nodes.append((x, y))

        node_positions.append(layer_nodes)

    def draw_layer_connections(weights, from_layer, to_layer):
        max_w = torch.max(torch.abs(weights)).item() + 1e-5
        out_dim, in_dim = weights.shape

        for j in range(in_dim):
            x1, y1 = node_positions[from_layer][j]
            for i in range(out_dim):
                x2, y2 = node_positions[to_layer][i]
                val = weights[i, j].item()

                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                angle = math.degrees(math.atan2(dy, dx))

                norm_weight = abs(val) / max_w
                thickness = 0.0005 + norm_weight * 0.0035
                edge_color = color.lime if val >= 0 else color.red

                Entity(
                    parent=graph_container,
                    model='quad',
                    position=((x1 + x2) / 2.0, (y1 + y2) / 2.0, 0.01),
                    scale=(length, thickness),
                    rotation_z=angle,
                    color=edge_color
                )

    draw_layer_connections(w1, 0, 1)
    draw_layer_connections(w2, 1, 2)
    draw_layer_connections(w3, 2, 3)

    for layer in node_positions:
        for (nx, ny) in layer:
            Entity(
                parent=graph_container,
                model='circle',
                color=color.azure,
                position=(nx, ny, -0.02),
                scale=(0.015, 0.015)
            )


# ---------------------------------------------------------------------------
# TOP VIEW RADAR VISUALIZER
# ---------------------------------------------------------------------------
def build_top_view_radar():
    for child in list(radar_container.children):
        destroy(child)

    Text(
        text="Top View (Tactical Radar)",
        parent=radar_container,
        position=(-0.25, 0.38, -0.02),
        scale=0.9,
        color=color.orange
    )

    # Radar frame and grid
    Entity(
        parent=radar_container,
        model='quad',
        color=color.rgba(20, 30, 45, 200),
        scale=(0.55, 0.55),
        position=(0, 0, 0.02)
    )

    # Grid crosshairs
    Entity(parent=radar_container, model='quad', color=color.dark_gray, scale=(0.53, 0.002), position=(0, 0, 0.01))
    Entity(parent=radar_container, model='quad', color=color.dark_gray, scale=(0.002, 0.53), position=(0, 0, 0.01))

    scale_factor = 0.0035  # UI distance per world unit

    # Helper function to make an arrow visual
    def create_arrow(parent, pos_x, pos_y, heading_deg, color_val, size_scale=1.0):
        arrow_root = Entity(
            parent=parent,
            position=(pos_x, pos_y, -0.02),
            rotation_z=-heading_deg
        )
        # Body
        Entity(parent=arrow_root, model='quad', color=color_val, scale=(0.006 * size_scale, 0.025 * size_scale), position=(0, -0.005, 0))
        # Arrowhead
        Entity(parent=arrow_root, model='quad', color=color_val, scale=(0.014 * size_scale, 0.014 * size_scale), position=(0, 0.01, 0), rotation_z=45)

    # 1. Main Player Plane (Orange Arrow)
    create_arrow(radar_container, 0, 0, airplane.rotation_y, color.orange, size_scale=1.2)

    # 2. Target Position (Red Dot)
    target_pos = wingmen[0]._formation_target()
    rel_target_x = (target_pos.x - airplane.x) * scale_factor
    rel_target_z = (target_pos.z - airplane.z) * scale_factor

    if abs(rel_target_x) < 0.25 and abs(rel_target_z) < 0.25:
        Entity(
            parent=radar_container,
            model='circle',
            color=color.red,
            position=(rel_target_x, rel_target_z, -0.02),
            scale=(0.02, 0.02)
        )

    # 3. Wingmen (Green Arrows)
    for w in wingmen:
        rel_x = (w.entity.x - airplane.x) * scale_factor
        rel_z = (w.entity.z - airplane.z) * scale_factor

        if abs(rel_x) < 0.25 and abs(rel_z) < 0.25:
            create_arrow(radar_container, rel_x, rel_z, w.entity.rotation_y, color.lime, size_scale=0.9)


def update_menu_status(message=None):
    ai_state_str = "FROZEN" if game_paused else "ACTIVE"
    msg_str = f" | [{message}]" if message else ""
    menu_status.text = f"Active Wingmen: {len(wingmen)}   |   Flight & AI Physics: {ai_state_str}{msg_str}"


def show_menu(msg=None):
    global game_paused
    game_paused = True  # Movement completely frozen
    menu_container.enabled = True
    update_menu_status(msg)
    build_network_graph()
    build_top_view_radar()


def hide_menu():
    global game_paused
    menu_container.enabled = False
    game_paused = False  # Resume movement


def toggle_menu():
    if menu_container.enabled:
        hide_menu()
    else:
        show_menu()


def input(key):
    global game_paused
    if key in ('p', 'm'):
        if menu_container.enabled:
            hide_menu()
        else:
            show_menu("GAME PAUSED")
    elif key == 'o':
        for w in wingmen:
            w.save()
        show_menu("BRAINS SAVED")
    elif key == 'i':
        for w in wingmen:
            w.load()
        show_menu("BRAINS LOADED")
    elif key == 'r':
        for w in wingmen:
            w.hard_reset()
        if menu_container.enabled:
            show_menu("RESET WINGMEN")


# ---------------------------------------------------------------------------
# GAME LOOP
# ---------------------------------------------------------------------------
def update():
    global speed, vertical_velocity

    # Movement and simulation physics completely frozen when paused or menu open
    if game_paused or menu_container.enabled:
        return

    # Ground tracking
    ground.x = airplane.x
    ground.z = airplane.z

    # Forward flight physics
    airplane.position += airplane.forward * speed * time.dt

    # Lift dynamics
    lift = (speed / 50) ** 2 * gravity
    vertical_velocity = lift - gravity

    if airplane.y > 0 or vertical_velocity > 0:
        airplane.y += vertical_velocity * time.dt
    else:
        airplane.y = 0

    # Controls
    pitch_input = 0
    roll_input = 0

    if held_keys['w']: pitch_input += turn_speed
    if held_keys['s']: pitch_input -= turn_speed
    if held_keys['a']: roll_input -= turn_speed
    if held_keys['d']: roll_input += turn_speed

    # Rotation
    airplane.rotate(Vec3(pitch_input * time.dt, 0, roll_input * time.dt))

    # Coordinated Yaw
    bank_angle = airplane.rotation_z
    airplane.rotate(Vec3(0, bank_angle * 0.8 * time.dt, 0))

    # Throttle
    if held_keys['space']: speed += 10 * time.dt
    if held_keys['left shift']: speed = max(15, speed - 10 * time.dt)

    # Update Wingmen
    for w in wingmen:
        w.update(wingmen)

    # HUD updates
    heading_deg = int(airplane.rotation_y % 360)
    altitude_ft = int(airplane.y)

    hud_text.text = (
        f"SPD : {speed:.0f} KTS\n"
        f"HDG : {heading_deg:03d}°\n"
        f"ALT : {altitude_ft} FT"
    )

    hud_horizon_pivot.rotation_z = -airplane.rotation_z
    pitch_offset = -airplane.rotation_x * 0.004
    hud_horizon_pivot.y = max(-0.35, min(0.35, pitch_offset))


app.run()