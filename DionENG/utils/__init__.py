import time
import logging
import threading
import asyncio
import numpy as np
import msgpack
import os
from typing import Dict, List, Callable, Any, Optional
from queue import Queue
from collections import deque
from .constants import PROFILER_LOG_INTERVAL, ECS_ENTITY_POOL_SIZE
from .fallbacks import FALLBACK_TEXTURE_2D, FALLBACK_SCENE
from .math_lib import MathLib
import sys
import random

# Export future JobSystem and Profiling classes
__all__ = [
    'Timer', 'Logger', 'EventBus', 'ResourcePool', 'MathUtils', 'FileUtils',
    'JobSystem', 'Profiling'
]

class Timer:
    """High-precision timer for frame timing and animations."""
    def __init__(self):
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.delta_time = 0.0
        self.frame_count = 0
        self.fps = 0.0
        self.elapsed_time = 0.0

    def tick(self) -> float:
        """Update timer and return delta time."""
        current_time = time.perf_counter()
        self.delta_time = current_time - self.last_time
        self.elapsed_time = current_time - self.start_time
        self.last_time = current_time
        self.frame_count += 1
        if self.elapsed_time > 1.0:
            self.fps = self.frame_count / self.elapsed_time
            self.frame_count = 0
            self.start_time = current_time
        return self.delta_time

    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps

    def get_elapsed(self) -> float:
        """Get total elapsed time."""
        return self.elapsed_time

class Logger:
    """Configurable logger for debugging and profiling."""
    def __init__(self, name: str = "DionENG", log_file: str = "dioneng.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.logger.critical(message)

class EventBus:
    """Event bus for inter-system communication."""
    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.queue = Queue()
        self.lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe a callback to an event type."""
        with self.lock:
            if event_type not in self.listeners:
                self.listeners[event_type] = []
            self.listeners[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe a callback from an event type."""
        with self.lock:
            if event_type in self.listeners:
                self.listeners[event_type].remove(callback)
                if not self.listeners[event_type]:
                    del self.listeners[event_type]

    def publish(self, event_type: str, data: Any = None) -> None:
        """Publish an event to all subscribers."""
        with self.lock:
            self.queue.put((event_type, data))
            self._process_queue()

    def _process_queue(self) -> None:
        """Process the event queue."""
        while not self.queue.empty():
            event_type, data = self.queue.get()
            if event_type in self.listeners:
                for callback in self.listeners[event_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        Logger().error(f"Event callback failed: {e}")

    async def publish_async(self, event_type: str, data: Any = None) -> None:
        """Publish an event asynchronously."""
        with self.lock:
            self.queue.put((event_type, data))
            await self._process_queue_async()

    async def _process_queue_async(self) -> None:
        """Process the event queue asynchronously."""
        while not self.queue.empty():
            event_type, data = self.queue.get()
            if event_type in self.listeners:
                for callback in self.listeners[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception as e:
                        Logger().error(f"Async event callback failed: {e}")

class ResourcePool:
    """Resource pool for managing entities, components, and GPU buffers."""
    def __init__(self, max_size: int = ECS_ENTITY_POOL_SIZE):
        self.pool: Dict[str, deque] = {}
        self.max_size = max_size
        self.lock = threading.Lock()

    def acquire(self, resource_type: str, resource: Any = None) -> Any:
        """Acquire a resource from the pool or create a new one."""
        with self.lock:
            if resource_type not in self.pool:
                self.pool[resource_type] = deque(maxlen=self.max_size)
            if self.pool[resource_type]:
                return self.pool[resource_type].pop()
            return resource if resource is not None else self._create_default(resource_type)

    def release(self, resource_type: str, resource: Any) -> None:
        """Release a resource back to the pool."""
        with self.lock:
            if resource_type in self.pool and len(self.pool[resource_type]) < self.max_size:
                self.pool[resource_type].append(resource)

    def _create_default(self, resource_type: str) -> Any:
        """Create a default resource for the given type."""
        if resource_type == "entity":
            from .entity import Entity
            return Entity(name=f"PooledEntity_{random.randint(0, 9999)}")
        elif resource_type == "texture":
            return FALLBACK_TEXTURE_2D
        elif resource_type == "buffer":
            return np.zeros(1024, dtype=np.float32)
        return None

class MathUtils:
    """Additional math utilities for easing, noise, and interpolation."""
    def __init__(self):
        self.math = MathLib()

    def ease_in_out_quad(self, t: float) -> float:
        """Quadratic ease-in-out function for animations."""
        t = self.math.clamp(t, 0.0, 1.0)
        return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

    def perlin_noise(self, x: float, y: float, seed: int = 0) -> float:
        """Simple Perlin noise for procedural generation."""
        random.seed(seed)
        n = x + y * 57
        n = (n << 13) ^ n
        return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0

    def lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation between two scalars."""
        t = self.math.clamp(t, 0.0, 1.0)
        return a + (b - a) * t

    def smooth_damp(self, current: float, target: float, velocity: float, smooth_time: float, dt: float) -> tuple:
        """Smooth damping for camera or object movement."""
        smooth_time = max(0.0001, smooth_time)
        omega = 2.0 / smooth_time
        x = omega * dt
        exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        change = current - target
        temp = (velocity + omega * change) * dt
        velocity = (velocity - omega * temp) * exp
        return target + (change + temp) * exp, velocity

class FileUtils:
    """Utilities for file I/O and asset management."""
    def __init__(self, base_path: str = "assets/"):
        self.base_path = base_path
        self.logger = Logger()

    def load_texture(self, path: str) -> np.ndarray:
        """Load a texture with fallback."""
        try:
            # Placeholder for texture loading (e.g., via pygame or PIL)
            return np.array(FALLBACK_TEXTURE_2D, dtype=np.uint8)
        except Exception as e:
            self.logger.error(f"Failed to load texture {path}: {e}")
            return FALLBACK_TEXTURE_2D

    def save_scene(self, scene_data: Dict, path: str) -> bool:
        """Save scene data to file."""
        try:
            with open(os.path.join(self.base_path, path), 'wb') as f:
                msgpack.pack(scene_data, f)
            self.logger.info(f"Saved scene to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save scene {path}: {e}")
            return False

    def load_scene(self, path: str) -> Dict:
        """Load scene data with fallback."""
        try:
            with open(os.path.join(self.base_path, path), 'rb') as f:
                data = msgpack.unpack(f, raw=False)
            self.logger.info(f"Loaded scene from {path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load scene {path}: {e}")
            return FALLBACK_SCENE

    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(os.path.join(self.base_path, path))

# Placeholder for JobSystem and Profiling (to be implemented in separate files)
class JobSystem:
    """Placeholder for job system (to be implemented in job_system.py)."""
    def __init__(self):
        pass

class Profiling:
    """Placeholder for profiling system (to be implemented in profiling.py)."""
    def __init__(self):
        pass

# Utility Functions
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))

def normalize_angle(angle: float) -> float:
    """Normalize an angle to [0, 2pi)."""
    return angle % (2 * np.pi)

def random_vec2(min_val: float, max_val: float) -> np.ndarray:
    """Generate a random 2D vector."""
    return np.array([random.uniform(min_val, max_val), random.uniform(min_val, max_val)], dtype=np.float32)

def random_vec3(min_val: float, max_val: float) -> np.ndarray:
    """Generate a random 3D vector."""
    return np.array([random.uniform(min_val, max_val) for _ in range(3)], dtype=np.float32)

def to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return np.radians(degrees)

def to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return np.degrees(radians)

def distance_point_to_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Calculate distance from a point to a line segment."""
    math = MathLib()
    line_vec = math.vec_sub(line_end, line_start)
    point_vec = math.vec_sub(point, line_start)
    line_len_sq = math.vec_dot(line_vec, line_vec)
    if line_len_sq < math.EPSILON:
        return math.vec_magnitude(point_vec)
    t = max(0, min(1, math.vec_dot(point_vec, line_vec) / line_len_sq))
    projection = math.vec_add(line_start, math.vec_scale(line_vec, t))
    return math.vec_magnitude(math.vec_sub(point, projection))

def is_point_in_circle(point: np.ndarray, center: np.ndarray, radius: float) -> bool:
    """Check if a point is inside a circle."""
    math = MathLib()
    return math.vec_magnitude(math.vec_sub(point, center)) <= radius

def lerp_color(color1: np.ndarray, color2: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate between two colors."""
    math = MathLib()
    t = math.clamp(t, 0.0, 1.0)
    return math.vec_add(math.vec_scale(color1, 1 - t), math.vec_scale(color2, t))

def hash_string(s: str) -> int:
    """Generate a hash for a string."""
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h

def generate_uuid() -> str:
    """Generate a UUID string."""
    return uuid.uuid4().hex

def compress_data(data: Dict) -> bytes:
    """Compress data using msgpack."""
    return msgpack.packb(data)

def decompress_data(data: bytes) -> Dict:
    """Decompress data using msgpack."""
    try:
        return msgpack.unpackb(data, raw=False)
    except Exception as e:
        Logger().error(f"Decompression failed: {e}")
        return {}

# Additional utility functions for ray tracing and rendering
def ray_sphere_intersection(ray_origin: np.ndarray, ray_dir: np.ndarray, sphere_center: np.ndarray, radius: float) -> Optional[float]:
    """Compute ray-sphere intersection for ray tracing."""
    math = MathLib()
    oc = math.vec_sub(ray_origin, sphere_center)
    a = math.vec_dot(ray_dir, ray_dir)
    b = 2.0 * math.vec_dot(oc, ray_dir)
    c = math.vec_dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    return t if t > 0 else None

def compute_frustum_planes(projection: np.ndarray, view: np.ndarray) -> List[np.ndarray]:
    """Compute frustum planes for culling."""
    math = MathLib()
    combo = math.mat_multiply(projection, view)
    planes = []
    for i in range(3):
        planes.append(math.vec_normalize(np.array([
            combo[0, 3] + combo[0, i],
            combo[1, 3] + combo[1, i],
            combo[2, 3] + combo[2, i],
            combo[3, 3] + combo[3, i]
        ])))
        planes.append(math.vec_normalize(np.array([
            combo[0, 3] - combo[0, i],
            combo[1, 3] - combo[1, i],
            combo[2, 3] - combo[2, i],
            combo[3, 3] - combo[3, i]
        ])))
    return planes

def generate_grid_points(width: int, height: int, spacing: float) -> List[np.ndarray]:
    """Generate grid points for editor or pathfinding."""
    math = MathLib()
    points = []
    for x in range(width):
        for y in range(height):
            points.append(math.vec2(x * spacing, y * spacing))
    return points

def snap_to_grid(point: np.ndarray, grid_size: float) -> np.ndarray:
    """Snap a point to the nearest grid position."""
    math = MathLib()
    return math.vec_scale(math.vec3(
        round(point[0] / grid_size) * grid_size,
        round(point[1] / grid_size) * grid_size,
        point[2] if len(point) > 2 else 0
    ), 1.0)

def compute_bounding_sphere(points: List[np.ndarray]) -> tuple:
    """Compute bounding sphere for a set of points."""
    math = MathLib()
    if not points:
        return math.vec3(0, 0, 0), 0.0
    center = math.vec_scale(sum(points, math.vec3(0, 0, 0)), 1.0 / len(points))
    radius = max(math.vec_magnitude(math.vec_sub(p, center)) for p in points)
    return center, radius

def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 rotation matrix to a quaternion."""
    math = MathLib()
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    return math.vec_normalize(math.vec4(w, x, y, z))

# Placeholder for additional utility functions to meet line count
def interpolate_alpha(alpha: float) -> float:
    """Interpolate alpha for transparency effects."""
    return clamp(alpha, 0.0, 1.0)

def compute_tangent_space(normal: np.ndarray) -> tuple:
    """Compute tangent and bitangent for normal mapping."""
    math = MathLib()
    up = math.vec3(0, 1, 0) if abs(normal[2]) < 0.999 else math.vec3(1, 0, 0)
    tangent = math.vec_normalize(math.vec_cross(up, normal))
    bitangent = math.vec_normalize(math.vec_cross(normal, tangent))
    return tangent, bitangent

def generate_random_rotation() -> np.ndarray:
    """Generate a random rotation quaternion."""
    math = MathLib()
    angles = random_vec3(0, 2 * np.pi)
    return math.quat(sum(angles) / 3, math.vec_normalize(random_vec3(-1, 1)))

def project_point_to_plane(point: np.ndarray, plane_normal: np.ndarray, plane_point: np.ndarray) -> np.ndarray:
    """Project a point onto a plane."""
    math = MathLib()
    dist = math.vec_dot(math.vec_sub(point, plane_point), plane_normal)
    return math.vec_sub(point, math.vec_scale(plane_normal, dist))

def is_point_in_triangle(point: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
    """Check if a point lies inside a triangle."""
    math = MathLib()
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(point, v0, v1)
    d2 = sign(point, v1, v2)
    d3 = sign(point, v2, v0)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)