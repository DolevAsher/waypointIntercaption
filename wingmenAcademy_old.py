import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim

from ursina import *
import random
import math
import time

app = Ursina()

# ---------------------------------------------------------------------------
# GLOBAL GAME & TIMER STATE
# ---------------------------------------------------------------------------
game_paused = True

cycle_timer = 0.0
cycle_duration = 20.0
auto_reset_enabled = True
auto_save_enabled = True
last_action_status = "Initial Startup"

DEFAULT_SAVE_PATH = "shared_wingman_brain.pt"

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
    color=color.orange,
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
turn_speed = 50
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
    text=" [M/P] Menu & Pause | [O] Save | [I] Load | [R] Reset Wingmen ",
    position=(-0.45, 0.48),
    scale=1.0,
    color=color.yellow,
    background=True
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
# SHARED NEURAL NETWORK
# ---------------------------------------------------------------------------
OBS_DIM = 13
HIDDEN_DIM = 32
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
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.2)

    def forward(self, x):
        h = self.trunk(x)
        mean = torch.tanh(self.mean_head(h))
        value = self.value_head(h)
        return mean, value


shared_net = PolicyNet()
shared_optimizer = optim.Adam(shared_net.parameters(), lr=5e-4)


def save_shared_weights(filepath=DEFAULT_SAVE_PATH):
    torch.save(shared_net.state_dict(), filepath)


def load_shared_weights(filepath=DEFAULT_SAVE_PATH):
    if os.path.exists(filepath):
        shared_net.load_state_dict(torch.load(filepath))
        return True
    return False


class WingmanAI:
    def __init__(self, leader, offset=Vec3(25, 0, -25), name="Wingman", body_color=None):
        self.leader = leader
        self.offset = offset
        self.name = name

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

        self.gamma = 0.95
        self.states = []
        self.actions = []
        self.rewards = []

        self.steps_per_update = 40
        self.step_count = 0
        self.episode_reward = 0.0
        self.paused = False

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

        with torch.no_grad():
            mean, _ = shared_net(state)
            std = shared_net.log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = mean if self.paused else dist.sample()
            action_clamped = torch.clamp(action, -1, 1)

        pitch_cmd, roll_cmd, throttle_cmd = action_clamped.tolist()

        self.rotation_x += pitch_cmd * turn_speed * time.dt
        self.rotation_z += roll_cmd * turn_speed * time.dt
        self.rotation_y += self.rotation_z * 0.8 * time.dt
        self.speed = max(15.0, min(120.0, self.speed + throttle_cmd * 15 * time.dt))

        self.entity.rotation = (self.rotation_x, self.rotation_y, self.rotation_z)
        self.entity.position += self.entity.forward * self.speed * time.dt

        lift = (self.speed / 50.0) ** 2 * gravity
        vert_vel = lift - gravity
        self.entity.y = max(0.5, self.entity.y + vert_vel * time.dt)

        dist_to_target = distance(self.entity.position, target)

        reward = 2.0 * math.exp(-dist_to_target / 15.0) - 0.1 * (dist_to_target / 50.0)
        reward -= (abs(pitch_cmd) + abs(roll_cmd)) * 0.02
        if min_peer_dist < 8.0:
            reward -= 2.0

        self.episode_reward += reward

        if not self.paused:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

        self.step_count += 1

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
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.step_count = 0
        self.episode_reward = 0.0

    def _learn(self, bootstrap=True):
        if not self.rewards:
            return

        states_tensor = torch.stack(self.states)
        actions_tensor = torch.stack(self.actions)

        means, values = shared_net(states_tensor)
        stds = shared_net.log_std.exp()
        dists = torch.distributions.Normal(means, stds)

        log_probs = dists.log_prob(actions_tensor).sum(dim=-1)
        entropies = dists.entropy().sum(dim=-1)
        values = values.squeeze(-1)

        with torch.no_grad():
            if bootstrap:
                last_state, _, _ = self.get_state([])
                _, next_value = shared_net(last_state)
                R = next_value.squeeze()
            else:
                R = torch.tensor(0.0)

        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = (returns - values).pow(2).mean()
        entropy_bonus = entropies.mean()

        loss = policy_loss + 0.5 * value_loss - 0.001 * entropy_bonus

        shared_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(shared_net.parameters(), 1.0)
        shared_optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.step_count = 0
        self.episode_reward = 0.0


# ---------------------------------------------------------------------------
# WINGMEN CREATION
# ---------------------------------------------------------------------------
def generate_right_wingmen(n):
    bots = []
    for i in range(n):
        hue = (i * 360 / max(1, n))
        bot_color = color.hsv(hue, 0.8, 0.9)

        staggered_offset = Vec3(25 + (i * 8), 0, -25 - (i * 12))

        bots.append(WingmanAI(
            leader=airplane,
            offset=staggered_offset,
            name=f"Bot-{i + 1}",
            body_color=bot_color
        ))
    return bots


NUM_WINGMEN = 10
wingmen = generate_right_wingmen(NUM_WINGMEN)

# ---------------------------------------------------------------------------
# MAIN MENU & OVERLAYS
# ---------------------------------------------------------------------------
menu_container = Entity(parent=camera.ui, enabled=True, position=(0, 0, -0.05))

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
    position=(-0.80, 0.38, -0.01),
    scale=0.9,
    color=color.yellow
)

timer_info_text = Text(
    text="",
    parent=menu_container,
    position=(-0.80, 0.33, -0.01),
    scale=0.9,
    color=color.lime
)

# Graph and Radar containers attached inside the menu container
graph_container = Entity(parent=menu_container, position=(-0.50, -0.05, -0.01), scale=0.82)
radar_container = Entity(parent=menu_container, position=(0.10, -0.05, -0.01))
file_selector_container = Entity(parent=menu_container, position=(0.60, 0.15, -0.01))


# MENU CONTROLS & ACTION BUTTONS
def btn_start_sim():
    hide_menu()


def btn_toggle_auto_reset():
    global auto_reset_enabled
    auto_reset_enabled = not auto_reset_enabled
    update_menu_status(f"Auto-Reset {'ENABLED' if auto_reset_enabled else 'DISABLED'}")


def btn_toggle_auto_save():
    global auto_save_enabled
    auto_save_enabled = not auto_save_enabled
    update_menu_status(f"Auto-Save {'ENABLED' if auto_save_enabled else 'DISABLED'}")


def btn_adjust_time(amt):
    global cycle_duration
    cycle_duration = max(5.0, cycle_duration + amt)
    update_menu_status(f"Cycle Limit set to {cycle_duration:.0f}s")


def btn_adjust_wingmen(amt):
    global NUM_WINGMEN, wingmen
    new_count = max(1, min(25, NUM_WINGMEN + amt))
    if new_count == NUM_WINGMEN:
        return

    NUM_WINGMEN = new_count
    for w in wingmen:
        destroy(w.entity)
        destroy(w.target_dot)

    wingmen = generate_right_wingmen(NUM_WINGMEN)
    update_menu_status(f"Wingmen count set to {NUM_WINGMEN}")
    build_top_view_radar()


def btn_manual_save():
    save_shared_weights()
    update_menu_status("SAVED SHARED WEIGHTS -> " + DEFAULT_SAVE_PATH)


def btn_manual_reset():
    global cycle_timer
    for w in wingmen:
        w.hard_reset()
    cycle_timer = 0.0
    update_menu_status("WINGMEN RESET")


# UI Control Buttons inside Menu
start_button = Button(text="[START / RESUME]", parent=menu_container, position=(0.60, 0.42, -0.02), scale=(0.28, 0.05),
                      color=color.azure, on_click=btn_start_sim)
save_button = Button(text="Save Weights", parent=menu_container, position=(0.60, 0.35, -0.02), scale=(0.18, 0.04),
                     color=color.blue, on_click=btn_manual_save)
reset_button = Button(text="Reset Wingmen", parent=menu_container, position=(0.60, 0.29, -0.02), scale=(0.18, 0.04),
                      color=color.red, on_click=btn_manual_reset)

time_minus_btn = Button(text="-5s", parent=menu_container, position=(0.15, 0.42, -0.02), scale=(0.06, 0.04),
                        color=color.dark_gray, on_click=lambda: btn_adjust_time(-5))
time_plus_btn = Button(text="+5s", parent=menu_container, position=(0.23, 0.42, -0.02), scale=(0.06, 0.04),
                       color=color.dark_gray, on_click=lambda: btn_adjust_time(5))
auto_reset_btn = Button(text="Toggle Auto-Reset", parent=menu_container, position=(0.19, 0.35, -0.02),
                        scale=(0.18, 0.04), color=color.dark_gray, on_click=btn_toggle_auto_reset)
auto_save_btn = Button(text="Toggle Auto-Save", parent=menu_container, position=(0.19, 0.29, -0.02), scale=(0.18, 0.04),
                       color=color.dark_gray, on_click=btn_toggle_auto_save)

wingmen_minus_btn = Button(text="-1 Bot", parent=menu_container, position=(-0.25, 0.42, -0.02), scale=(0.07, 0.04),
                           color=color.dark_gray, on_click=lambda: btn_adjust_wingmen(-1))
wingmen_plus_btn = Button(text="+1 Bot", parent=menu_container, position=(-0.17, 0.42, -0.02), scale=(0.07, 0.04),
                          color=color.dark_gray, on_click=lambda: btn_adjust_wingmen(1))


# FILE SELECTION DIALOG IN MENU
def build_file_selection_slot():
    for child in list(file_selector_container.children):
        destroy(child)

    Text(
        text="Load Weight File:",
        parent=file_selector_container,
        position=(-0.15, 0.12, -0.02),
        scale=0.8,
        color=color.cyan
    )

    weight_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    if not weight_files:
        weight_files = [DEFAULT_SAVE_PATH]

    for idx, fname in enumerate(weight_files[:3]):
        y_pos = 0.05 - (idx * 0.05)

        def make_load_handler(file_to_load):
            return lambda: load_file_action(file_to_load)

        Button(
            text=fname[:20],
            parent=file_selector_container,
            position=(0, y_pos, -0.02),
            scale=(0.30, 0.04),
            color=color.dark_gray,
            on_click=make_load_handler(fname)
        )


def load_file_action(filename):
    if load_shared_weights(filename):
        update_menu_status(f"LOADED WEIGHTS: {filename}")
    else:
        update_menu_status(f"ERROR: {filename} not found")


# LIVE GRAPH VISUALIZER (Standard Colors, No Flashing)
def build_network_graph():
    for child in list(graph_container.children):
        destroy(child)

    Text(
        text="Live Policy Network Weights",
        parent=graph_container,
        position=(-0.35, 0.38, -0.02),
        scale=0.9,
        color=color.lime
    )

    w1 = shared_net.trunk[0].weight.data
    w2 = shared_net.trunk[2].weight.data
    w3 = shared_net.mean_head.weight.data

    layers = [OBS_DIM, HIDDEN_DIM, HIDDEN_DIM, ACT_DIM]
    x_positions = [-0.18, -0.02, 0.14, 0.30]

    node_positions = []
    for l_idx, num_nodes in enumerate(layers):
        x = x_positions[l_idx]
        layer_nodes = []
        y_span = 0.50
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
                thickness = 0.0005 + norm_weight * 0.003
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
                scale=(0.012, 0.012)
            )

    input_label_desc = "INPUTS (13):\n• Target Rel\n• Pitch/Roll\n• Delta Spd/Alt\n• Leader Pos\n• Peer Prox"
    Text(text=input_label_desc, parent=graph_container, position=(-0.42, 0.22, -0.02), scale=0.65, color=color.cyan)

    output_label_desc = "OUTPUTS (3):\n• Pitch Cmd\n• Roll Cmd\n• Throttle"
    Text(text=output_label_desc, parent=graph_container, position=(0.33, 0.10, -0.02), scale=0.65, color=color.yellow)


# TOP VIEW RADAR VISUALIZER
def build_top_view_radar():
    for child in list(radar_container.children):
        destroy(child)

    Text(
        text="Tactical Radar Overview",
        parent=radar_container,
        position=(-0.25, 0.38, -0.02),
        scale=0.9,
        color=color.orange
    )

    Entity(
        parent=radar_container,
        model='quad',
        color=color.rgba(20, 30, 45, 200),
        scale=(0.42, 0.42),
        position=(0, -0.05, 0.02)
    )

    Entity(parent=radar_container, model='quad', color=color.dark_gray, scale=(0.40, 0.002), position=(0, -0.05, 0.01))
    Entity(parent=radar_container, model='quad', color=color.dark_gray, scale=(0.002, 0.40), position=(0, -0.05, 0.01))

    all_positions = [airplane.position] + [w.entity.position for w in wingmen]
    if wingmen:
        all_positions.append(wingmen[0]._formation_target())

    max_dx = max(abs(p.x - airplane.x) for p in all_positions)
    max_dz = max(abs(p.z - airplane.z) for p in all_positions)
    max_offset = max(max_dx, max_dz, 50.0)

    dynamic_scale_factor = 0.18 / max_offset

    def create_arrow(parent, pos_x, pos_y, heading_deg, color_val, size_scale=1.0):
        arrow_root = Entity(
            parent=parent,
            position=(pos_x, pos_y - 0.05, -0.02),
            rotation_z=-heading_deg
        )
        Entity(parent=arrow_root, model='quad', color=color_val, scale=(0.006 * size_scale, 0.025 * size_scale),
               position=(0, -0.005, 0))
        Entity(parent=arrow_root, model='quad', color=color_val, scale=(0.014 * size_scale, 0.014 * size_scale),
               position=(0, 0.01, 0), rotation_z=45)

    create_arrow(radar_container, 0, 0, airplane.rotation_y, color.orange, size_scale=1.2)

    if wingmen:
        target_pos = wingmen[0]._formation_target()
        rel_target_x = (target_pos.x - airplane.x) * dynamic_scale_factor
        rel_target_z = (target_pos.z - airplane.z) * dynamic_scale_factor

        Entity(
            parent=radar_container,
            model='circle',
            color=color.red,
            position=(rel_target_x, rel_target_z - 0.05, -0.02),
            scale=(0.02, 0.02)
        )

    for w in wingmen:
        rel_x = (w.entity.x - airplane.x) * dynamic_scale_factor
        rel_z = (w.entity.z - airplane.z) * dynamic_scale_factor
        create_arrow(radar_container, rel_x, rel_z, w.entity.rotation_y, color.lime, size_scale=0.9)


def update_menu_status(message=None):
    global last_action_status
    if message:
        last_action_status = message

    ai_state_str = "PAUSED" if game_paused else "RUNNING"
    menu_status.text = f"Sim State: {ai_state_str}   |   Status: {last_action_status}"
    timer_info_text.text = (
        f"Cycle Timer: {cycle_timer:.1f}s / {cycle_duration:.0f}s   |   "
        f"Wingmen: {NUM_WINGMEN}   |   "
        f"Auto-Reset: {'ON' if auto_reset_enabled else 'OFF'}   |   "
        f"Auto-Save: {'ON' if auto_save_enabled else 'OFF'}"
    )


def show_menu(msg=None):
    global game_paused
    game_paused = True
    menu_container.enabled = True
    update_menu_status(msg)
    build_network_graph()
    build_top_view_radar()
    build_file_selection_slot()


def hide_menu():
    global game_paused
    menu_container.enabled = False
    game_paused = False


def toggle_menu():
    if menu_container.enabled:
        hide_menu()
    else:
        show_menu("GAME PAUSED")


def input(key):
    global game_paused
    if key in ('p', 'm'):
        toggle_menu()
    elif key == 'o':
        btn_manual_save()
    elif key == 'i':
        load_shared_weights()
        update_menu_status("DEFAULT WEIGHTS LOADED")
    elif key == 'r':
        btn_manual_reset()


# ---------------------------------------------------------------------------
# GAME LOOP
# ---------------------------------------------------------------------------
def update():
    global speed, vertical_velocity, cycle_timer

    if menu_container.enabled:
        update_menu_status()
        return

    if not game_paused:
        cycle_timer += time.dt

        if auto_reset_enabled and cycle_timer >= cycle_duration:
            if auto_save_enabled:
                save_shared_weights()
                update_menu_status("AUTO-SAVED & RESET CYCLE")
            else:
                update_menu_status("AUTO-RESET CYCLE")

            for w in wingmen:
                w.hard_reset()
            cycle_timer = 0.0

    ground.x = airplane.x
    ground.z = airplane.z

    airplane.position += airplane.forward * speed * time.dt

    lift = (speed / 50) ** 2 * gravity
    vertical_velocity = lift - gravity

    if airplane.y > 0 or vertical_velocity > 0:
        airplane.y += vertical_velocity * time.dt
    else:
        airplane.y = 0

    pitch_input = 0
    roll_input = 0

    if held_keys['w']: pitch_input += turn_speed
    if held_keys['s']: pitch_input -= turn_speed
    if held_keys['a']: roll_input -= turn_speed
    if held_keys['d']: roll_input += turn_speed

    airplane.rotate(Vec3(pitch_input * time.dt, 0, roll_input * time.dt))

    bank_angle = airplane.rotation_z
    airplane.rotate(Vec3(0, bank_angle * 0.8 * time.dt, 0))

    if held_keys['space']: speed += 10 * time.dt
    if held_keys['left shift']: speed = max(15, speed - 10 * time.dt)

    for w in wingmen:
        w.update(wingmen)

    heading_deg = int(airplane.rotation_y % 360)
    altitude_ft = int(airplane.y)

    hud_text.text = (
        f"SPD : {speed:.0f} KTS\n"
        f"HDG : {heading_deg:03d}°\n"
        f"ALT : {altitude_ft} FT\n"
        f"CYC : {cycle_timer:.1f}s / {cycle_duration:.0f}s"
    )

    hud_horizon_pivot.rotation_z = -airplane.rotation_z
    pitch_offset = -airplane.rotation_x * 0.004
    hud_horizon_pivot.y = max(-0.35, min(0.35, pitch_offset))


show_menu("PAUSED ON STARTUP")

app.run()