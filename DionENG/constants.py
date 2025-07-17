import pygame
import moderngl
import pyopencl as cl
import numpy as np
import msgpack
import comtypes.client
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from watchdog.observers import Observer

# Window Settings
WINDOW_SIZE = (1280, 720)
WINDOW_TITLE = "DionENG"
FULLSCREEN = False
TARGET_FPS = 60
MIN_DT = 1.0 / 240.0

# Rendering Constants
RENDER_QUALITY_DEFAULT = 1.0
RENDER_QUALITY_MIN = 0.5
RENDER_QUALITY_MAX = 2.0
POST_EFFECTS = {
    'bloom': False,
    'ssao': False,
    'motion_blur': False,
    'ray_tracing': False
}
SHADOW_MAP_SIZE = (1024, 1024)
OPENGL_VERSION = (4, 3)
OPENCL_PLATFORM = 0
OPENCL_DEVICE = 0
PARTICLE_COUNT_MAX = 1000
BATCH_SIZE_MAX = 100

# Physics Constants
GRAVITY_2D = (0, -9.81)
GRAVITY_3D = (0, -9.81, 0)
COLLISION_LAYERS = {
    'player': 1,
    'enemy': 2,
    'environment': 4,
    'projectile': 8
}
QUADTREE_BOUNDS = [0, 0, 1000, 1000]
QUADTREE_MAX_DEPTH = 8
QUADTREE_MAX_OBJECTS = 10

# Input Constants
INPUT_MAPPINGS = {
    'move_forward': pygame.K_w,
    'move_backward': pygame.K_s,
    'move_left': pygame.K_a,
    'move_right': pygame.K_d,
    'jump': pygame.K_SPACE,
    'interact': pygame.K_e,
    'toggle_editor': pygame.K_e
}
GAMEPAD_BUTTONS = {
    'jump': 0,
    'interact': 1
}
INPUT_BUFFER_SIZE = 100

# Networking Constants
NETWORK_PORT = 5555
NETWORK_PROTOCOL_DEFAULT = 'udp'
NETWORK_RPC_IDS = {
    'update_position': 1,
    'update_health': 2
}
NETWORK_DELTA_THRESHOLD = 0.01
NETWORK_LAG_COMPENSATION = 0.1

# AI Constants
AI_PATHFINDING_GRID_SIZE = (100, 100)
AI_FSM_STATES = {
    'idle': 0,
    'patrol': 1,
    'chase': 2,
    'attack': 3
}
AI_VISIBILITY_RANGE = 50.0
AI_PATHFINDING_MAX_NODES = 1000

# UI Constants
UI_WIDGET_TYPES = {
    'label': 'label',
    'button': 'button',
    'progress_bar': 'progress_bar'
}
UI_ANIMATION_DURATIONS = {
    'fade': 0.5,
    'slide': 0.3
}
UI_FONT_DEFAULT = 'arial'
UI_FONT_SIZE_DEFAULT = 24

# Audio Constants
AUDIO_3D_RANGE = 50.0
AUDIO_DOPPLER_FACTOR = 1.0
AUDIO_MAX_CHANNELS = 32
AUDIO_MUSIC_FADE_TIME = 2.0

# Editor Constants
EDITOR_GRID_SIZE = 1.0
EDITOR_SNAP_VALUE = 0.5
EDITOR_MAX_UNDO = 50

# Profiling Constants
PROFILER_METRIC_THRESHOLD = {
    'fps': 30,
    'update_time': 0.016,
    'render_time': 0.016
}
PROFILER_LOG_INTERVAL = 5.0

# Job System Constants
JOB_SYSTEM_MAX_THREADS = 8
JOB_SYSTEM_TASK_PRIORITIES = {
    'physics': 1,
    'rendering': 2,
    'ai': 3,
    'network': 4
}

# Entity-Component System Constants
ECS_MAX_ENTITIES = 10000
ECS_MAX_COMPONENTS = 100000
ECS_ENTITY_POOL_SIZE = 1000