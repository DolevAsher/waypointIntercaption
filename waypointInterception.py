import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from ursina import *
import random
import math
import time
import copy
import tkinter as tk
from tkinter import filedialog
import json

app = Ursina()

# ---------------------------------------------------------------------------
# GLOBAL STATE & CONFIGURATION
# ---------------------------------------------------------------------------
game_paused = True
auto_evolve = False

# Set to an int for reproducible runs, or None for a fresh random run each launch.
RANDOM_SEED = 42
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

cycle_timer = 0.0
cycle_duration = 10.0
SIM_SPEED_MULTIPLIER = 1.0  # simulated seconds per real second - fast-forwards training without changing flight physics
population_size = 100
mutation_rate = 0.05
mutation_strength = 0.15
elite_genetic_percentage = 0.5

generation_count = 0
generation_metrics = []
global_next_id = 1

# --- SPATIAL & WAYPOINT GLOBALS ---
START_POS = (0.0, 200.0, 0.0)
NUM_WAYPOINTS = 3
WAYPOINT_RADIUS = 30.0
CONSTANT_SPEED = 60.0
WAYPOINT_CLEAR_BONUS = 30000.0  # subtracted from accumulated distance when a waypoint is cleared
EARLY_ARRIVAL_BONUS_SCALE = 1.0  # extra bonus multiplier for reaching a waypoint early in the cycle (0 = no time credit)

# --- FLIGHT MODEL RATES ---
PITCH_RATE = 50.0   # deg/sec at full pitch command
ROLL_RATE = 50.0    # deg/sec at full roll command
YAW_COUPLING = 0.8  # how strongly roll induces yaw

waypoints_t = None
waypoint_entities = []

last_cycle_survivors = set()
last_population = []

# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------
Sky()
ground = Entity(model='plane', scale=10000, texture='grass', texture_scale=(1000, 1000), y=0)

start_balloon = Entity(model='sphere', color=color.green, scale=15, unlit=True, position=START_POS)
Text(parent=start_balloon, text="0", scale=5, position=(0, 0.6, 0), color=color.white, billboard=True, origin=(0, 0))


# ---------------------------------------------------------------------------
# COURSE GENERATION
# ---------------------------------------------------------------------------
def generate_course():
    global waypoints_t, waypoint_entities

    for w in waypoint_entities: destroy(w)
    waypoint_entities.clear()

    pts = []
    current_base = torch.tensor(START_POS, dtype=torch.float32)

    waypoint_hues = [0, 30, 60, 300, 330]

    for i in range(NUM_WAYPOINTS):
        t_x = current_base[0] + random.uniform(-150, 150)
        t_z = current_base[2] + random.uniform(100, 225)
        t_y = max(30.0, current_base[1] + random.uniform(-50, 50))

        pt = torch.tensor([t_x, t_y, t_z])
        pts.append(pt)
        current_base = pt

        ring_color = color.hsv(waypoint_hues[i % len(waypoint_hues)], 0.9, 0.9)
        ent = Entity(model='sphere', color=ring_color, scale=WAYPOINT_RADIUS * 2, unlit=True, position=(t_x, t_y, t_z))
        Text(parent=ent, text=f"{i + 1}", scale=5, position=(0, 0.6, 0), color=color.white, billboard=True,
             origin=(0, 0))
        waypoint_entities.append(ent)

    waypoints_t = torch.stack(pts)


# ---------------------------------------------------------------------------
# SPECTATOR CAMERA & HUD
# ---------------------------------------------------------------------------
def frame_spectator_camera():
    """
    Position/aim the spectator camera so the whole generated course (the
    start balloon plus every waypoint) fits in view. Waypoint positions are
    randomized per course, so a fixed camera offset only happened to frame
    some layouts and clipped others - this instead frames whatever course
    was actually generated.
    """
    if waypoints_t is None:
        return

    all_pts = torch.cat([torch.tensor([START_POS], dtype=torch.float32), waypoints_t], dim=0)
    min_xyz = all_pts.min(dim=0).values
    max_xyz = all_pts.max(dim=0).values
    center = (min_xyz + max_xyz) / 2.0

    # Widest horizontal extent of the course, with a floor so a very short/
    # straight course doesn't leave the camera sitting awkwardly close.
    span = max((max_xyz[0] - min_xyz[0]).item(), (max_xyz[2] - min_xyz[2]).item(), 200.0)

    cam_x = center[0].item() + span * 0.9
    cam_y = center[1].item()  # same altitude as the waypoints - horizontal view, no downward tilt
    cam_z = min_xyz[2].item() - span * 0.6  # pull back behind the earliest point in the course

    camera.position = (cam_x, cam_y, cam_z)
    camera.look_at((center[0].item(), center[1].item(), center[2].item()))
    camera.rotation_z = 0  # look_at can introduce roll; force level (horizontal) framing


# Fallback framing before a course exists yet (waypoints_t is still None here).
# frame_spectator_camera() is called again once initialize_population() has
# generated the real course, and re-called whenever the course is randomized.
camera.position = (START_POS[0] + 800, START_POS[1] + 250, START_POS[2] + 400)
camera.look_at((START_POS[0], START_POS[1], START_POS[2] + 400))
camera.rotation_z = 0
hud_text = Text(position=(-0.85, 0.45), scale=1.5, color=color.green)

# ---------------------------------------------------------------------------
# NEURAL NETWORK (GENOME) - TRIGONOMETRIC GUIDANCE INPUTS
# ---------------------------------------------------------------------------
OBS_DIM = 7  # Alignment Cos, Azimuth Error, Elevation Error, Distance, Pitch, Roll, Altitude
HIDDEN_DIM = 24
ACT_DIM = 2  # Pitch & Roll commands


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


class BatchedPolicyNet:
    """
    Runs every genome's forward pass in a single vectorized batch instead of
    looping over the population and calling set_weights_flat() + forward()
    once per bot. This is the same math as PolicyNet, just executed as one
    batched matmul per layer (shape (P, out, in) @ (P, in, 1)) rather than
    population_size sequential single-genome forward passes.

    Only valid for a PolicyNet whose trunk is Linear/Tanh/Linear/Tanh/Linear/Tanh,
    since it hardcodes 3 linear layers pulled from named_parameters() order.
    """

    def __init__(self, reference_net):
        self.param_shapes = [p.shape for p in reference_net.parameters()]
        self.param_sizes = [p.numel() for p in reference_net.parameters()]

    def forward(self, batch_weights, obs):
        # batch_weights: (P, total_params), obs: (P, obs_dim)
        w0, b0, w2, b2, w4, b4 = torch.split(batch_weights, self.param_sizes, dim=1)
        P = batch_weights.shape[0]

        w0 = w0.view(P, *self.param_shapes[0])
        w2 = w2.view(P, *self.param_shapes[2])
        w4 = w4.view(P, *self.param_shapes[4])

        x = obs.unsqueeze(-1)  # (P, obs_dim, 1)
        x = torch.tanh(torch.bmm(w0, x).squeeze(-1) + b0)
        x = torch.tanh(torch.bmm(w2, x.unsqueeze(-1)).squeeze(-1) + b2)
        x = torch.tanh(torch.bmm(w4, x.unsqueeze(-1)).squeeze(-1) + b4)
        return x  # (P, act_dim)


class Genome:
    def __init__(self, weights, bot_id, age=0):
        self.weights = weights
        self.id = bot_id
        self.age = age
        self.fitness = 0.0
        self.history = []

        safe_hue = 180 + ((bot_id * 47) % 90)
        self.color = color.hsv(safe_hue, 0.9, 0.9)


# ---------------------------------------------------------------------------
# POPULATION & BATCH STATE
# ---------------------------------------------------------------------------
population = []
bot_pos = None
bot_rot = None

bot_target_idx = None
bot_cleared_count = None
bot_min_dist = None
bot_accumulated_dist = None

visual_bots = []
shared_net = PolicyNet()
batched_net = BatchedPolicyNet(shared_net)


def initialize_population(size):
    global population, global_next_id, last_population
    generate_course()
    base_net = PolicyNet()
    population = []
    for _ in range(size):
        w = base_net.get_weights_flat() + torch.randn_like(base_net.get_weights_flat()) * 0.5
        population.append(Genome(w, global_next_id))
        global_next_id += 1

    last_population = list(population)
    reset_batch_states()


def mutate_weights(weights):
    mutated = weights.clone()
    mask = torch.rand_like(mutated) < mutation_rate
    mutated[mask] += torch.randn_like(mutated[mask]) * mutation_strength
    return mutated


def reset_batch_states():
    global bot_pos, bot_rot, population_size
    global bot_target_idx, bot_cleared_count, bot_min_dist, bot_accumulated_dist

    bot_pos = torch.zeros((population_size, 3))
    bot_pos[:, 0] = START_POS[0]
    bot_pos[:, 1] = START_POS[1]
    bot_pos[:, 2] = START_POS[2]

    bot_rot = torch.zeros((population_size, 3))

    bot_target_idx = torch.zeros(population_size, dtype=torch.long)
    bot_cleared_count = torch.zeros(population_size, dtype=torch.float32)
    bot_min_dist = torch.full((population_size,), float('inf'))
    bot_accumulated_dist = torch.zeros(population_size, dtype=torch.float32)

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
    global population_size, population, global_next_id, last_population
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
    last_population = list(population)
    setup_visual_bots()
    reset_batch_states()


# ---------------------------------------------------------------------------
# CYCLE COMPLETION & SELECTION
# ---------------------------------------------------------------------------
def finish_cycle():
    global generation_count, population, global_next_id, cycle_timer, last_cycle_survivors, last_population

    for i in range(population_size):
        population[i].fitness = bot_accumulated_dist[i].item()

    population.sort(key=lambda x: x.fitness)

    generation_metrics.append({
        'min': population[0].fitness,
        'max': population[-1].fitness,
        'mean': sum(p.fitness for p in population) / population_size
    })

    generation_count += 1

    for p in population:
        p.age += 1

    last_population = list(population)

    weights = [population_size - i for i in range(population_size)]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    last_cycle_survivors.clear()

    num_survivors = max(1, int(population_size * elite_genetic_percentage))
    chosen_indices = random.choices(range(population_size), weights=probabilities, k=num_survivors)

    if 0 not in chosen_indices:
        chosen_indices[0] = 0

    # chosen_indices may repeat (weighted sampling with replacement) - deep-copy
    # each survivor so duplicate slots never share one Genome/weights-tensor
    # instance (that aliasing previously meant "different" bots could secretly
    # mutate/report state for each other).
    surviving_parents = [copy.deepcopy(population[idx]) for idx in chosen_indices]

    new_population = []

    for p in surviving_parents:
        new_population.append(p)
        last_cycle_survivors.add(p.id)

    chosen_ids_str = ", ".join(str(p.id) for p in new_population)

    while len(new_population) < population_size:
        parent = random.choice(surviving_parents)
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

menu_title = Text(text="SEQUENTIAL WAYPOINT GA", parent=menu_container, position=(-0.88, 0.44, -0.01), scale=2.0,
                  color=color.cyan)
timer_info_text = Text(text="", parent=menu_container, position=(-0.88, 0.38, -0.01), scale=1.5, color=color.lime)

Button(text="[ RESUME CYCLE ]", parent=menu_container, position=(-0.71, 0.30, -0.02), scale=(0.28, 0.045),
       color=color.azure, text_color=color.white, on_click=lambda: hide_menu())


class Checkbox(Button):
    def __init__(self, label="AUTO EVOLVE", value=False, on_change=None, **kwargs):
        self.label_text = label
        self.value = value
        self.on_change = on_change
        kwargs['text'] = f"[{'X' if self.value else ' '}] {self.label_text}"

        if 'color' not in kwargs: kwargs['color'] = color.violet
        if 'text_color' not in kwargs: kwargs['text_color'] = color.white
        if 'highlight_color' not in kwargs: kwargs['highlight_color'] = color.violet.tint(0.2)
        super().__init__(**kwargs)
        self.on_click = self.toggle

    def toggle(self):
        self.value = not self.value
        self.text = f"[{'X' if self.value else ' '}] {self.label_text}"
        if self.on_change:
            self.on_change(self.value)


auto_evolve_cb = Checkbox(label="AUTO EVOLVE", parent=menu_container, position=(-0.35, 0.30, -0.02),
                          scale=(0.28, 0.045), on_change=lambda val: globals().update(auto_evolve=val))


def randomize_and_reset():
    generate_course()
    reset_batch_states()
    frame_spectator_camera()
    timer_info_text.text = f"Cycle: {generation_count} (Course Reset)"


Button(text="[ RANDOMIZE COURSE ]", parent=menu_container, position=(-0.35, 0.23, -0.02), scale=(0.28, 0.045),
       color=color.red.tint(-0.2), text_color=color.white, on_click=randomize_and_reset)


def open_load_dialog():
    global last_population
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

            last_population = list(population)
            print(f"Successfully injected weights from {os.path.basename(filepath)}")
            build_bot_board()
            for j, ent in enumerate(visual_bots):
                if j < len(population): ent.color = population[j].color
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

Text(text="Population Size:", parent=menu_container, position=(-0.15, 0.44, -0.01), scale=1.4, color=color.black)
pop_val_text = Text(text="", parent=menu_container, position=(0.00, 0.34, -0.01), scale=1.4, color=color.black)
Button(text="-", parent=menu_container, position=(0.11, 0.33, -0.02), scale=(0.035, 0.045), color=color.red,
       on_click=lambda: change_pop(-10))
Button(text="+", parent=menu_container, position=(0.15, 0.33, -0.02), scale=(0.035, 0.045), color=color.green,
       on_click=lambda: change_pop(10))

Text(text="Cycle Duration:", parent=menu_container, position=(0.19, 0.44, -0.01), scale=1.4, color=color.black)
cycle_val_text = Text(text="", parent=menu_container, position=(0.33, 0.34, -0.01), scale=1.4, color=color.black)
Button(text="-", parent=menu_container, position=(0.44, 0.33, -0.02), scale=(0.035, 0.045), color=color.red,
       on_click=lambda: change_cycle_time(-1.0))
Button(text="+", parent=menu_container, position=(0.48, 0.33, -0.02), scale=(0.035, 0.045), color=color.green,
       on_click=lambda: change_cycle_time(1.0))

Text(text="Sim Speed:", parent=menu_container, position=(0.53, 0.44, -0.01), scale=1.4, color=color.black)
sim_speed_val_text = Text(text="", parent=menu_container, position=(0.65, 0.34, -0.01), scale=1.4, color=color.black)
Button(text="-", parent=menu_container, position=(0.715, 0.33, -0.02), scale=(0.035, 0.045), color=color.red,
       on_click=lambda: change_sim_speed(-1.0))
Button(text="+", parent=menu_container, position=(0.755, 0.33, -0.02), scale=(0.035, 0.045), color=color.green,
       on_click=lambda: change_sim_speed(1.0))


# --- SPECTATOR CONTROLS DIAGRAM ---
# Image lives at textures/spectator_controls.png, next to this script - Ursina's
# default asset search picks up images placed in a "textures" folder that sits
# alongside the entry-point .py file, so it's referenced here by name only.
controls_image = Entity(parent=menu_container, model='quad', texture='spectator_controls',
                        scale=(0.42, 0.56), position=(0.71, 0.01, -0.02))


def update_param_labels():
    pop_val_text.text = str(population_size)
    cycle_val_text.text = f"{cycle_duration:.1f}s"
    sim_speed_val_text.text = f"{SIM_SPEED_MULTIPLIER:.1f}x"


def change_pop(delta):
    update_population_size(max(10, min(500, population_size + delta)))
    update_param_labels()
    build_bot_board()


def change_cycle_time(delta):
    global cycle_duration
    cycle_duration = max(2.0, min(30.0, cycle_duration + delta))
    update_param_labels()


def change_sim_speed(delta):
    global SIM_SPEED_MULTIPLIER
    SIM_SPEED_MULTIPLIER = max(1.0, min(20.0, SIM_SPEED_MULTIPLIER + delta))
    update_param_labels()


# --- HOVER TOOLTIP & BOT BOARD ---
tooltip_panel = Entity(parent=camera.ui, model='quad', color=color.rgba(0, 0, 0, 240), scale=(0.46, 0.32),
                       enabled=False, z=-0.1)
tt_text = Text(parent=tooltip_panel, position=(-0.46, 0.42, -0.01), scale=2.6, color=color.white)
tt_graph_bg = Entity(parent=tooltip_panel, model='quad', color=color.rgba(25, 35, 45, 255), scale=(0.88, 0.42),
                     position=(0, -0.16, -0.01))
tt_graph_line = Entity(parent=tt_graph_bg, position=(-0.5, -0.5, -0.01), scale=(1, 1))
Text(text="Accum. Dist vs Time", parent=tt_graph_bg, position=(-0.45, 0.42, -0.01), scale=2.6, color=color.orange)


class BotIcon(Button):
    def __init__(self, genome, is_survivor=False, rank=1, **kwargs):
        super().__init__(**kwargs)
        self.genome = genome
        self.is_survivor = is_survivor
        self.rank = rank
        self.jet_body = Entity(parent=self, model='cube', color=self.genome.color, scale=(0.35, 0.15, 0.55),
                               position=(0, 0, -0.01))
        self.jet_wings = Entity(parent=self.jet_body, model='cube', color=color.cyan, scale=(1.8, 0.1, 0.3),
                                position=(0, 0, -0.1))

    def on_mouse_enter(self):
        tooltip_panel.enabled = True
        clamped_x = max(-0.5, min(0.3, mouse.x + 0.2))
        clamped_y = max(-0.3, min(0.3, mouse.y - 0.12))
        tooltip_panel.position = (clamped_x, clamped_y)

        status = "Survivor / Parent" if self.is_survivor else "Eliminated"
        tt_text.text = f"Rank: #{self.rank} | Bot ID: {self.genome.id} | Age: {self.genome.age}\nStatus: {status}\nFitness (Sum): {self.genome.fitness:,.1f}"

        if self.genome.history:
            verts = []
            min_val = min(self.genome.history)
            max_val = max(self.genome.history)
            val_range = max_val - min_val if max_val != min_val else 1.0

            for idx, d in enumerate(self.genome.history):
                nx = idx / max(1, len(self.genome.history) - 1)
                ny = (d - min_val) / val_range
                verts.append(Vec3(nx, ny, -0.01))
            if len(verts) > 1:
                tt_graph_line.model = Mesh(vertices=verts, mode='line', thickness=3)
                tt_graph_line.color = color.lime if self.is_survivor else color.red

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

    display_pop = last_population if last_population else population
    sorted_pop = sorted(display_pop, key=lambda x: x.fitness)

    for i, genome in enumerate(sorted_pop):
        x = (i % cols) * spacing_x
        y = -(i // cols) * spacing_y

        is_survivor = genome.id in last_cycle_survivors
        bg_color = color.green.tint(-0.2) if is_survivor else color.red.tint(-0.4)

        icon = BotIcon(genome, is_survivor=is_survivor, rank=i + 1, parent=grid_parent, position=(x, y),
                       scale=(0.038, 0.038),
                       color=bg_color)
        bot_icons.append(icon)


# --- MULTI-GENERATION METRICS GRAPH ---
graph_bg = Entity(parent=menu_container, model='quad', color=color.rgba(15, 22, 32, 230), scale=(0.87, 0.48),
                  position=(0.03, -0.12, -0.01))
graph_line_min = Entity(parent=graph_bg, position=(-0.42, -0.38, -0.02))
graph_line_max = Entity(parent=graph_bg, position=(-0.42, -0.38, -0.02))
graph_line_mean = Entity(parent=graph_bg, position=(-0.42, -0.38, -0.02))
Text(text="Fitness Spread Across Cycles\n(Lower is Better)", parent=graph_bg, position=(-0.45, 0.44, -0.02), scale=2,
     color=color.black)
Text(text="Min (Lime) | Mean (Yellow) | Max (Red)", parent=graph_bg, position=(-0.45, -0.44, -0.02), scale=1.5,
     color=color.black)


def build_multi_gen_graph():
    if not generation_metrics: return
    n_gens = len(generation_metrics)

    all_mins = [m['min'] for m in generation_metrics]
    all_maxs = [m['max'] for m in generation_metrics]
    g_min = min(all_mins)
    g_max = max(all_maxs)

    v_range = g_max - g_min
    if v_range == 0: v_range = 1.0

    verts_min, verts_max, verts_mean = [], [], []
    for i, m in enumerate(generation_metrics):
        nx = (i / max(1, n_gens - 1)) * 0.84

        ny_min = ((m['min'] - g_min) / v_range) * 0.75
        ny_max = ((m['max'] - g_min) / v_range) * 0.75
        ny_mean = ((m['mean'] - g_min) / v_range) * 0.75

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


def input(key):
    if key in ('p', 'm'):
        show_menu() if not menu_container.enabled else hide_menu()


# ---------------------------------------------------------------------------
# INITIALIZATION & MAIN UPDATE LOOP
# ---------------------------------------------------------------------------
initialize_population(population_size)
setup_visual_bots()
frame_spectator_camera()
update_param_labels()
show_menu()


def update():
    global cycle_timer, bot_pos, bot_rot
    global bot_target_idx, bot_cleared_count, bot_min_dist, bot_accumulated_dist

    dt = time.dt

    cam_speed = 300.0
    cam_rot_speed = 90.0

    fwd_dir = Vec3(camera.forward.x, 0, camera.forward.z)
    if fwd_dir.length() > 0.001:
        fwd_dir.normalize()
    else:
        fwd_dir = Vec3(camera.up.x, 0, camera.up.z).normalized()

    rgt_dir = Vec3(camera.right.x, 0, camera.right.z)
    if rgt_dir.length() > 0.001:
        rgt_dir.normalize()
    else:
        rgt_dir = Vec3(1, 0, 0)

    if held_keys['w']: camera.position += fwd_dir * cam_speed * dt
    if held_keys['s']: camera.position -= fwd_dir * cam_speed * dt
    if held_keys['d']: camera.position += rgt_dir * cam_speed * dt
    if held_keys['a']: camera.position -= rgt_dir * cam_speed * dt

    if held_keys['space']: camera.y += cam_speed * dt
    if held_keys['shift']: camera.y -= cam_speed * dt

    if held_keys['up arrow']: camera.rotation_x -= cam_rot_speed * dt
    if held_keys['down arrow']: camera.rotation_x += cam_rot_speed * dt
    if held_keys['left arrow']: camera.rotation_y -= cam_rot_speed * dt
    if held_keys['right arrow']: camera.rotation_y += cam_rot_speed * dt

    if menu_container.enabled: return

    sim_dt = dt * SIM_SPEED_MULTIPLIER
    cycle_timer += sim_dt

    # Neural Network Observations - Trigonometric Guidance Inputs (7 Dimensions)
    # Vectorized across the whole population instead of looping bot-by-bot.
    with torch.no_grad():
        b_yaw = torch.deg2rad(bot_rot[:, 1])
        b_pitch = torch.deg2rad(bot_rot[:, 0])
        fwd_vec = torch.stack([
            torch.sin(b_yaw) * torch.cos(b_pitch),
            -torch.sin(b_pitch),
            torch.cos(b_yaw) * torch.cos(b_pitch),
        ], dim=1)  # (P, 3)

        t_pos = waypoints_t[bot_target_idx]  # (P, 3)
        rel = t_pos - bot_pos
        dist_to_target = rel.norm(dim=1)  # (P,)
        target_dir = rel / (dist_to_target.unsqueeze(1) + 1e-6)

        # 1. Trigonometric alignment cosine (-1 to 1)
        alignment_cos = (fwd_vec * target_dir).sum(dim=1)

        # 2. Horizontal azimuth angle error, wrapped to (-180, 180]
        heading_to_target = torch.rad2deg(torch.atan2(rel[:, 0], rel[:, 2]))
        azimuth_diff = (heading_to_target - bot_rot[:, 1] + 180) % 360 - 180

        # 3. Vertical elevation angle error, wrapped to (-90, 90] so it can't
        #    exceed the [-1, 1] input range once divided by 90 below (the
        #    original mod-360 wrap allowed up to +/-180 here).
        horiz_dist = torch.sqrt(rel[:, 0] ** 2 + rel[:, 2] ** 2)
        target_pitch = torch.rad2deg(torch.atan2(-rel[:, 1], horiz_dist))
        elevation_diff = (target_pitch - bot_rot[:, 0] + 90) % 180 - 90

        batch_obs = torch.stack([
            alignment_cos,                                  # 3D heading alignment (cos angle)
            azimuth_diff / 180.0,                            # Horizontal steering error
            elevation_diff / 90.0,                            # Vertical steering error
            torch.clamp(dist_to_target / 1000.0, max=1.0),   # Normalized distance
            bot_rot[:, 0] / 90.0,                             # Current pitch
            bot_rot[:, 2] / 90.0,                             # Current roll
            bot_pos[:, 1] / 500.0,                            # Current altitude
        ], dim=1).float()  # (P, 7)

        # One batched forward pass for the whole population instead of
        # population_size individual set_weights_flat() + forward() calls.
        batch_weights = torch.stack([g.weights for g in population])
        actions = batched_net.forward(batch_weights, batch_obs)  # (P, 2)

        # Vectorized physics update (constant speed, pitch & roll from actions)
        pitch_cmd = actions[:, 0]
        roll_cmd = actions[:, 1]

        bot_rot[:, 0] += pitch_cmd * PITCH_RATE * sim_dt
        bot_rot[:, 2] += roll_cmd * ROLL_RATE * sim_dt
        bot_rot[:, 1] += bot_rot[:, 2] * YAW_COUPLING * sim_dt

        b_yaw = torch.deg2rad(bot_rot[:, 1])
        b_pitch = torch.deg2rad(bot_rot[:, 0])
        fwd_vec = torch.stack([
            torch.sin(b_yaw) * torch.cos(b_pitch),
            -torch.sin(b_pitch),
            torch.cos(b_yaw) * torch.cos(b_pitch),
        ], dim=1)

        bot_pos += fwd_vec * CONSTANT_SPEED * sim_dt
        bot_pos[:, 1].clamp_(min=0.0)

        active_waypoint_pos = waypoints_t[bot_target_idx]
        dist = torch.norm(bot_pos - active_waypoint_pos, dim=1)

        bot_accumulated_dist += dist * sim_dt
        bot_min_dist = torch.minimum(bot_min_dist, dist)

        reached = dist < WAYPOINT_RADIUS
        has_next_waypoint = bot_target_idx < NUM_WAYPOINTS - 1
        advancing = reached & has_next_waypoint
        finishing = reached & ~has_next_waypoint & (bot_cleared_count < NUM_WAYPOINTS)

        # Reaching a waypoint early in the cycle earns a bigger bonus than
        # reaching it right before time runs out (1x base bonus at the very
        # end of the cycle, up to (1 + EARLY_ARRIVAL_BONUS_SCALE)x at t=0).
        time_credit_frac = max(0.0, 1.0 - (cycle_timer / cycle_duration))
        waypoint_bonus = WAYPOINT_CLEAR_BONUS * (1.0 + EARLY_ARRIVAL_BONUS_SCALE * time_credit_frac)

        bot_target_idx[advancing] += 1
        bot_cleared_count[advancing] += 1
        bot_accumulated_dist[advancing] -= waypoint_bonus
        bot_min_dist[advancing] = float('inf')

        bot_cleared_count[finishing] += 1
        bot_accumulated_dist[finishing] -= waypoint_bonus
        bot_min_dist[finishing] = 0.0

    # Per-bot history logging stays a Python loop (cheap list append, not math),
    # needed for the per-bot tooltip graphs in the UI.
    if int(cycle_timer * 10) > len(population[0].history if population else []):
        accum_list = bot_accumulated_dist.tolist()
        for i in range(population_size):
            population[i].history.append(accum_list[i])

    for i in range(min(10, population_size)):
        visual_bots[i].position = (bot_pos[i][0].item(), bot_pos[i][1].item(), bot_pos[i][2].item())
        visual_bots[i].rotation = (bot_rot[i][0].item(), bot_rot[i][1].item(), bot_rot[i][2].item())
        visual_bots[i].color = population[i].color

    hud_text.text = f"SPECTATOR MODE\nSWARM POPULATION: {population_size}\nGEN: {generation_count}\nCYCLE: {cycle_timer:.1f} / {cycle_duration:.0f}s"

    if cycle_timer >= cycle_duration:
        finish_cycle()


app.run()