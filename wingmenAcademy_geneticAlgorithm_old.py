import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from ursina import *
import random
import math
import time
import tkinter as tk
from tkinter import filedialog
import json

app = Ursina()

# ---------------------------------------------------------------------------
# GLOBAL STATE & CONFIGURATION
# ---------------------------------------------------------------------------
game_paused = True
auto_evolve = False

cycle_timer = 0.0
cycle_duration = 6.0
population_size = 100
mutation_rate = 0.05
mutation_strength = 0.15

generation_count = 0
generation_metrics = []

global_next_id = 1

# ---------------------------------------------------------------------------
# ENVIRONMENT & LEADER AIRPLANE
# ---------------------------------------------------------------------------
Sky()
ground = Entity(model='plane', scale=10000, texture='grass', texture_scale=(1000, 1000), y=0)

airplane = Entity(position=(0, 10, 0))
body = Entity(parent=airplane, model='cube', color=color.orange, scale=(3, 1, 5))
wings = Entity(parent=airplane, model='cube', color=color.blue, scale=(12, 0.1, 1.2), position=(0, 0, 0))

speed = 50.0
turn_speed = 50.0
vertical_velocity = 0.0
gravity = 9.8

camera.parent = airplane
camera.position = (0, 50, -220)
camera.rotation = (15, 0, 0)

global_target_dot = Entity(parent=airplane, model='sphere', color=color.red, scale=1.5, position=(25, 0, -65),
                           unlit=True, enabled=False)

# ---------------------------------------------------------------------------
# FLIGHT HUD
# ---------------------------------------------------------------------------
hud_text = Text(position=(-0.85, 0.45), scale=1.5, color=color.green)
hud_horizon_pivot = Entity(parent=camera.ui, position=(0, 0))
horizon_line = Entity(parent=hud_horizon_pivot, model='quad', color=color.green, scale=(0.35, 0.003))

# ---------------------------------------------------------------------------
# NEURAL NETWORK (GENOME)
# ---------------------------------------------------------------------------
OBS_DIM = 12
HIDDEN_DIM = 24
ACT_DIM = 3


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.trunk(x)

    def get_weights_flat(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_weights_flat(self, flat_weights):
        pointer = 0
        for p in self.parameters():
            num_el = p.numel()
            p.data.copy_(flat_weights[pointer:pointer + num_el].view(p.shape))
            pointer += num_el


class Genome:
    def __init__(self, weights, bot_id, age=0):
        self.weights = weights
        self.id = bot_id
        self.age = age
        self.fitness = 0.0
        self.history = []
        self.color = color.hsv((bot_id * 47) % 360, 0.9, 0.9)


# ---------------------------------------------------------------------------
# POPULATION & BATCH STATE
# ---------------------------------------------------------------------------
population = []
bot_pos = None
bot_rot = None
bot_speed = None
cumulative_distances = None

visual_bots = []
shared_net = PolicyNet()


def initialize_population(size):
    global population, global_next_id
    base_net = PolicyNet()
    population = []
    for _ in range(size):
        w = base_net.get_weights_flat() + torch.randn_like(base_net.get_weights_flat()) * 0.5
        population.append(Genome(w, global_next_id))
        global_next_id += 1
    reset_batch_states()


# Best practice Continuous Control Mutation: Pure Gaussian perturbations
def mutate_weights(weights):
    mutated = weights.clone()
    mask = torch.rand_like(mutated) < mutation_rate
    mutated[mask] += torch.randn_like(mutated[mask]) * mutation_strength
    return mutated


def reset_batch_states():
    global bot_pos, bot_rot, bot_speed, cumulative_distances, population_size

    # Calculate target position dynamically based on leader airplane's pose
    lead_pos_t = torch.tensor([airplane.x, airplane.y, airplane.z], dtype=torch.float32)
    lead_yaw = airplane.rotation_y
    offset_t = torch.tensor([25.0, 0.0, -65.0], dtype=torch.float32)

    yaw_rad = math.radians(lead_yaw)
    rot_x = offset_t[0] * math.cos(yaw_rad) + offset_t[2] * math.sin(yaw_rad)
    rot_z = -offset_t[0] * math.sin(yaw_rad) + offset_t[2] * math.cos(yaw_rad)
    target = lead_pos_t + torch.tensor([rot_x, offset_t[1], rot_z])

    # Initialize bot positions to start right at the target location
    bot_pos = target.repeat(population_size, 1)

    bot_rot = torch.zeros((population_size, 3))
    bot_rot[:, 1] = airplane.rotation_y
    bot_speed = torch.full((population_size,), speed)
    cumulative_distances = torch.zeros(population_size)
    for p in population:
        p.history = []

    for i, ent in enumerate(visual_bots):
        if i < len(population):
            ent.color = population[i].color


def setup_visual_bots():
    global visual_bots
    for b in visual_bots: destroy(b)
    visual_bots = []
    for i in range(min(10, population_size)):
        ent = Entity(model='cube', scale=(3, 1, 5))
        Entity(parent=ent, model='cube', color=color.white50, scale=(4, 0.1, 0.24))
        visual_bots.append(ent)


def update_population_size(new_size):
    global population_size, population, global_next_id
    if new_size == population_size: return

    base_net = PolicyNet()
    if new_size > population_size:
        for _ in range(new_size - population_size):
            w = base_net.get_weights_flat() + torch.randn_like(base_net.get_weights_flat()) * 0.5
            population.append(Genome(w, global_next_id))
            global_next_id += 1
    else:
        population = population[:new_size]

    population_size = new_size
    setup_visual_bots()
    reset_batch_states()


# ---------------------------------------------------------------------------
# CYCLE COMPLETION & SELECTION
# ---------------------------------------------------------------------------
def finish_cycle():
    global generation_count, population, global_next_id, cycle_timer

    for i in range(population_size):
        population[i].fitness = cumulative_distances[i].item()

    # Sort population by fitness (lower is better)
    population.sort(key=lambda x: x.fitness)

    generation_metrics.append({
        'min': population[0].fitness,
        'max': population[-1].fitness,
        'mean': sum(p.fitness for p in population) / population_size
    })

    generation_count += 1

    # Strict Elitism: Keep only the Top 20%
    elite_count = max(1, int(population_size * 0.20))
    survivors = population[:elite_count]

    for s in survivors: s.age += 1

    new_population = list(survivors)

    # 4 Mutated children per elite survivor
    for parent in survivors:
        for _ in range(4):
            if len(new_population) >= population_size:
                break
            child_w = mutate_weights(parent.weights)
            new_population.append(Genome(child_w, global_next_id, age=0))
            global_next_id += 1

    # Fill any remaining slots (rounding artifacts) with random survivor mutations
    while len(new_population) < population_size:
        parent = random.choice(survivors)
        child_w = mutate_weights(parent.weights)
        new_population.append(Genome(child_w, global_next_id, age=0))
        global_next_id += 1

    population = new_population
    cycle_timer = 0.0
    reset_batch_states()

    if not auto_evolve:
        show_menu()


# ---------------------------------------------------------------------------
# UI - DASHBOARD
# ---------------------------------------------------------------------------
menu_container = Entity(parent=camera.ui, enabled=True, position=(0, 0, -0.05))
menu_bg = Entity(parent=menu_container, model='quad', color=color.rgba(10, 15, 25, 245), scale=(1.85, 0.95),
                 position=(0, 0, 0))

menu_title = Text(text="GENETIC WINGMEN ACADEMY", parent=menu_container, position=(-0.88, 0.44, -0.01), scale=2.0,
                  color=color.cyan)
timer_info_text = Text(text="", parent=menu_container, position=(-0.88, 0.38, -0.01), scale=1.5, color=color.lime)

Button(text="[ RESUME CYCLE ]", parent=menu_container, position=(-0.71, 0.30, -0.02), scale=(0.28, 0.045),
       color=color.azure, text_color=color.white, on_click=lambda: hide_menu())


class Checkbox(Button):
    def __init__(self, label="Auto Evolve", value=False, on_change=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.on_change = on_change
        self.lbl = Text(text=f"[{'X' if self.value else ' '}] {label}", parent=self, position=(-0.42, 0.15, -0.01),
                        scale=3.0, color=color.yellow)
        self.color = color.rgba(30, 40, 50, 200)
        self.on_click = self.toggle

    def toggle(self):
        self.value = not self.value
        self.lbl.text = f"[{'X' if self.value else ' '}] Auto Evolve"
        if self.on_change:
            self.on_change(self.value)


auto_evolve_cb = Checkbox(parent=menu_container, position=(-0.35, 0.30, -0.02), scale=(0.28, 0.045),
                          on_change=lambda val: globals().update(auto_evolve=val))


# --- LOAD & SAVE WEIGHTS (JSON FORMAT) ---
def open_load_dialog():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select Weights File", filetypes=[("JSON Files", "*.json")])
    if filepath:
        try:
            with open(filepath, 'r') as f:
                loaded_weights = json.load(f)

            for i, w_list in enumerate(loaded_weights):
                if i < len(population):
                    population[i].weights = torch.tensor(w_list, dtype=torch.float32)
                    population[i].age = 0

            print(f"Successfully injected weights from {os.path.basename(filepath)}")
            build_bot_board()

            for j, ent in enumerate(visual_bots):
                if j < len(population):
                    ent.color = population[j].color

        except Exception as e:
            print(f"Failed to load weights: {e}")


def save_weights_dialog():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    filepath = filedialog.asksaveasfilename(title="Save Top Weights", defaultextension=".json",
                                            filetypes=[("JSON Files", "*.json")])
    if filepath:
        try:
            top_weights = [p.weights.cpu().tolist() for p in population[:10]]
            with open(filepath, 'w') as f:
                json.dump(top_weights, f)
            print(f"Successfully saved weights to {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Failed to save weights: {e}")


Button(text="[ LOAD WEIGHTS ]", parent=menu_container, position=(-0.71, 0.23, -0.02), scale=(0.28, 0.045),
       color=color.orange, text_color=color.black, on_click=open_load_dialog)

Button(text="[ SAVE WEIGHTS ]", parent=menu_container, position=(-0.71, 0.16, -0.02), scale=(0.28, 0.045),
       color=color.lime, text_color=color.black, on_click=save_weights_dialog)

# --- PARAMETER CONTROLS ---
Text(text="Population Size:", parent=menu_container, position=(0.0, 0.44, -0.01), scale=1.4, color=color.white)
pop_val_text = Text(text="", parent=menu_container, position=(0.14, 0.44, -0.01), scale=1.4, color=color.yellow)
Button(text="-", parent=menu_container, position=(0.22, 0.43, -0.02), scale=(0.035, 0.045), color=color.red,
       on_click=lambda: change_pop(-10))
Button(text="+", parent=menu_container, position=(0.26, 0.43, -0.02), scale=(0.035, 0.045), color=color.green,
       on_click=lambda: change_pop(10))

Text(text="Cycle Duration:", parent=menu_container, position=(0.33, 0.44, -0.01), scale=1.4, color=color.white)
cycle_val_text = Text(text="", parent=menu_container, position=(0.48, 0.44, -0.01), scale=1.4, color=color.yellow)
Button(text="-", parent=menu_container, position=(0.58, 0.43, -0.02), scale=(0.035, 0.045), color=color.red,
       on_click=lambda: change_cycle_time(-1.0))
Button(text="+", parent=menu_container, position=(0.62, 0.43, -0.02), scale=(0.035, 0.045), color=color.green,
       on_click=lambda: change_cycle_time(1.0))


def update_param_labels():
    pop_val_text.text = str(population_size)
    cycle_val_text.text = f"{cycle_duration:.1f}s"


def change_pop(delta):
    update_population_size(max(10, min(500, population_size + delta)))
    update_param_labels()
    build_bot_board()


def change_cycle_time(delta):
    global cycle_duration
    cycle_duration = max(2.0, min(30.0, cycle_duration + delta))
    update_param_labels()


# --- HOVER TOOLTIP & BOT BOARD ---
tooltip_panel = Entity(parent=camera.ui, model='quad', color=color.rgba(0, 0, 0, 240), scale=(0.42, 0.32),
                       enabled=False, z=-0.1)
tt_text = Text(parent=tooltip_panel, position=(-0.45, 0.42, -0.01), scale=3.2, color=color.white)
tt_graph_bg = Entity(parent=tooltip_panel, model='quad', color=color.rgba(25, 35, 45, 255), scale=(0.85, 0.45),
                     position=(0, -0.15, -0.01))
tt_graph_line = Entity(parent=tt_graph_bg, position=(-0.5, -0.5, -0.01), scale=(1, 1))
Text(text="Distance vs Time", parent=tt_graph_bg, position=(-0.45, 0.42, -0.01), scale=3.0, color=color.orange)


class BotIcon(Button):
    def __init__(self, genome, is_elite=False, **kwargs):
        super().__init__(**kwargs)
        self.genome = genome
        self.is_elite = is_elite

        self.jet_body = Entity(parent=self, model='cube', color=self.genome.color, scale=(0.35, 0.15, 0.55),
                               position=(0, 0, -0.01))
        self.jet_wings = Entity(parent=self.jet_body, model='cube', color=color.cyan, scale=(1.8, 0.1, 0.3),
                                position=(0, 0, -0.1))

    def on_mouse_enter(self):
        tooltip_panel.enabled = True
        tooltip_panel.position = (mouse.x + 0.24, mouse.y - 0.12)
        status = "Elite" if self.is_elite else "Eliminated"
        tt_text.text = f"Bot ID: {self.genome.id} | Age: {self.genome.age}\nStatus: {status}\nAccumulated Dist: {self.genome.fitness:.1f}"

        if self.genome.history:
            verts = []
            max_d = max(self.genome.history) if max(self.genome.history) > 0 else 1.0
            for idx, d in enumerate(self.genome.history):
                nx = idx / max(1, len(self.genome.history) - 1)
                ny = min(1.0, max(0.0, d / max_d))
                verts.append(Vec3(nx, ny, -0.01))
            if len(verts) > 1:
                tt_graph_line.model = Mesh(vertices=verts, mode='line', thickness=3)
                tt_graph_line.color = color.lime if self.is_elite else color.red

    def on_mouse_exit(self):
        tooltip_panel.enabled = False


grid_parent = Entity(parent=menu_container, position=(-0.85, 0.12, -0.01))
bot_icons = []


def build_bot_board():
    for icon in bot_icons: destroy(icon)
    bot_icons.clear()

    cols = 10
    spacing_x = 0.042
    spacing_y = 0.042

    elite_count = max(1, int(population_size * 0.20))

    for i, genome in enumerate(population):
        x = (i % cols) * spacing_x
        y = -(i // cols) * spacing_y

        is_elite = i < elite_count
        bg_color = color.green.tint(-0.2) if is_elite else color.red.tint(-0.4)

        icon = BotIcon(genome, is_elite=is_elite, parent=grid_parent, position=(x, y), scale=(0.038, 0.038),
                       color=bg_color)
        bot_icons.append(icon)


# --- MULTI-GENERATION METRICS GRAPH ---
graph_bg = Entity(parent=menu_container, model='quad', color=color.rgba(15, 22, 32, 230), scale=(0.82, 0.45),
                  position=(0.42, -0.12, -0.01))
graph_line_min = Entity(parent=graph_bg, position=(-0.42, -0.38, -0.02))
graph_line_max = Entity(parent=graph_bg, position=(-0.42, -0.38, -0.02))
graph_line_mean = Entity(parent=graph_bg, position=(-0.42, -0.38, -0.02))

zero_line = Entity(parent=graph_bg, model='quad', color=color.rgba(255, 255, 255, 120), scale=(0.84, 0.003),
                   position=(0, -0.38, -0.01))
Text(text="Zero Dist (0.0)", parent=zero_line, position=(-0.48, 2.0, -0.01), scale=0.6, color=color.light_gray)

Text(text="Distance Spread Across Cycles", parent=graph_bg, position=(-0.45, 0.44, -0.02), scale=1.2, color=color.cyan)
Text(text="Min (Lime) | Mean (Yellow) | Max (Red)", parent=graph_bg, position=(0.02, 0.44, -0.02), scale=0.9,
     color=color.white)
Text(text="Cycle Number ->", parent=graph_bg, position=(0.25, -0.44, -0.02), scale=0.9, color=color.gray)


def build_multi_gen_graph():
    if not generation_metrics: return

    n_gens = len(generation_metrics)
    max_val = max(m['max'] for m in generation_metrics) or 1.0

    verts_min, verts_max, verts_mean = [], [], []

    for i, m in enumerate(generation_metrics):
        nx = (i / max(1, n_gens - 1)) * 0.84
        ny_min = (m['min'] / max_val) * 0.75
        ny_max = (m['max'] / max_val) * 0.75
        ny_mean = (m['mean'] / max_val) * 0.75

        verts_min.append(Vec3(nx, ny_min, -0.01))
        verts_max.append(Vec3(nx, ny_max, -0.01))
        verts_mean.append(Vec3(nx, ny_mean, -0.01))

    if len(verts_min) > 1:
        graph_line_min.model = Mesh(vertices=verts_min, mode='line', thickness=2)
        graph_line_min.color = color.lime
        graph_line_max.model = Mesh(vertices=verts_max, mode='line', thickness=2)
        graph_line_max.color = color.red
        graph_line_mean.model = Mesh(vertices=verts_mean, mode='line', thickness=2)
        graph_line_mean.color = color.yellow


def show_menu():
    global game_paused
    game_paused = True
    menu_container.enabled = True
    timer_info_text.text = f"Cycle: {generation_count}"
    update_param_labels()
    build_bot_board()
    build_multi_gen_graph()


def hide_menu():
    global game_paused
    game_paused = False
    menu_container.enabled = False
    global_target_dot.enabled = True


def input(key):
    if key in ('p', 'm'):
        show_menu() if not menu_container.enabled else hide_menu()


# ---------------------------------------------------------------------------
# INITIALIZATION & MAIN UPDATE LOOP
# ---------------------------------------------------------------------------
initialize_population(population_size)
setup_visual_bots()
update_param_labels()
show_menu()


def update():
    global speed, vertical_velocity, cycle_timer, bot_pos, bot_rot, bot_speed, cumulative_distances

    if menu_container.enabled: return

    dt = time.dt
    cycle_timer += dt

    # 1. Leader Flight Physics
    pitch_input = turn_speed if held_keys['w'] else -turn_speed if held_keys['s'] else 0
    roll_input = turn_speed if held_keys['d'] else -turn_speed if held_keys['a'] else 0

    airplane.rotate(Vec3(pitch_input * dt, 0, roll_input * dt))
    airplane.rotate(Vec3(0, airplane.rotation_z * 0.8 * dt, 0))

    effective_pitch = -airplane.rotation_x
    lift_factor = math.sin(math.radians(max(0.0, effective_pitch))) / math.sin(math.radians(8.0))
    vertical_velocity = (speed / 50) ** 2 * gravity * lift_factor - gravity
    airplane.y = max(0, airplane.y + vertical_velocity * dt)
    airplane.position += airplane.forward * speed * dt
    ground.x, ground.z = airplane.x, airplane.z

    if held_keys['space']: speed += 10 * dt
    if held_keys['left shift']: speed = max(15, speed - 10 * dt)

    # 2. PyTorch Batched Neural Evaluation
    lead_pos_t = torch.tensor([airplane.x, airplane.y, airplane.z], dtype=torch.float32)
    lead_yaw = airplane.rotation_y
    offset_t = torch.tensor([25.0, 0.0, -65.0], dtype=torch.float32)

    yaw_rad = math.radians(lead_yaw)
    rot_x = offset_t[0] * math.cos(yaw_rad) + offset_t[2] * math.sin(yaw_rad)
    rot_z = -offset_t[0] * math.sin(yaw_rad) + offset_t[2] * math.cos(yaw_rad)
    target = lead_pos_t + torch.tensor([rot_x, offset_t[1], rot_z])

    obs_list = []
    for i in range(population_size):
        rel = target - bot_pos[i]
        heading_diff = (lead_yaw - bot_rot[i, 1] + 180) % 360 - 180
        obs = torch.tensor([
            rel[0] / 50.0, rel[1] / 50.0, rel[2] / 50.0,
            bot_rot[i, 0] / 90.0, bot_rot[i, 2] / 90.0, heading_diff / 180.0,
            bot_rot[i, 1] / 180.0, lead_yaw / 180.0,
            (bot_speed[i] - 50.0) / 50.0, bot_pos[i, 1] / 100.0,
            lead_pos_t[0] / 500.0, lead_pos_t[2] / 1500.0
        ], dtype=torch.float32)
        obs_list.append(obs)

    batch_obs = torch.stack(obs_list)
    actions = []

    for i in range(population_size):
        shared_net.set_weights_flat(population[i].weights)
        with torch.no_grad():
            act = shared_net(batch_obs[i])
            actions.append(act)

    actions = torch.stack(actions)

    # 3. Batched Physics Update
    for i in range(population_size):
        pitch_cmd, roll_cmd, throttle_cmd = actions[i][0].item(), actions[i][1].item(), actions[i][2].item()

        bot_rot[i, 0] += pitch_cmd * turn_speed * dt
        bot_rot[i, 2] += roll_cmd * turn_speed * dt
        bot_rot[i, 1] += bot_rot[i, 2] * 0.8 * dt
        bot_speed[i] = max(15.0, min(120.0, bot_speed[i] + throttle_cmd * 15 * dt))

        b_yaw, b_pitch = math.radians(bot_rot[i, 1].item()), math.radians(bot_rot[i, 0].item())
        fx = math.sin(b_yaw) * math.cos(b_pitch)
        fy = -math.sin(b_pitch)
        fz = math.cos(b_yaw) * math.cos(b_pitch)

        bot_pos[i] += torch.tensor([fx, fy, fz]) * bot_speed[i] * dt

        dist = torch.dist(bot_pos[i], target)
        cumulative_distances[i] += dist * dt

        if int(cycle_timer * 10) > len(population[i].history):
            population[i].history.append(dist.item())

    # 4. Sync Visual Render Entities (Top 10)
    for i in range(min(10, population_size)):
        visual_bots[i].position = (bot_pos[i][0].item(), bot_pos[i][1].item(), bot_pos[i][2].item())
        visual_bots[i].rotation = (bot_rot[i][0].item(), bot_rot[i][1].item(), bot_rot[i][2].item())
        visual_bots[i].color = population[i].color

    hud_text.text = f"SPD: {speed:.0f} KTS\nHDG: {int(airplane.rotation_y % 360):03d}°\nALT: {int(airplane.y)} FT\nCYCLE: {cycle_timer:.1f} / {cycle_duration:.0f}s"
    hud_horizon_pivot.rotation_z = -airplane.rotation_z
    hud_horizon_pivot.y = max(-0.35, min(0.35, -airplane.rotation_x * 0.004))

    # 5. Cycle Timer Completion
    if cycle_timer >= cycle_duration:
        finish_cycle()


app.run()