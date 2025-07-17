import numpy as np
from .math_lib import MathLib
from .constants import COLLISION_LAYERS, AI_FSM_STATES, AI_VISIBILITY_RANGE
from .fallbacks import (
    FALLBACK_TEXTURE_2D, FALLBACK_CUBE_VERTICES, FALLBACK_CUBE_INDICES,
    FALLBACK_PLAYER_SCRIPT, FALLBACK_ENEMY_SCRIPT
)

class Component:
    """Base class for all components in DionENG."""
    def __init__(self, component_type, **kwargs):
        self.type = component_type
        self.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f"{self.type}(enabled={self.enabled})"

class Transform(Component):
    """Component for position, rotation, and scale with hierarchical support."""
    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0), scale=1.0, parent=None):
        super().__init__("Transform")
        self.math = MathLib()
        self.position = self.math.vec3(*position)
        self.rotation = self.math.quat(0, self.math.vec3(1, 0, 0))  # Identity quaternion
        self.scale = scale if isinstance(scale, (int, float)) else self.math.vec3(*scale)
        self.parent = parent
        self.local_matrix = self._compute_local_matrix()
        self.world_matrix = self._compute_world_matrix()

    def _compute_local_matrix(self):
        """Compute local transformation matrix."""
        scale_matrix = self.math.mat_scale(self.scale)
        rot_matrix = self.math.quat_to_matrix(self.rotation)
        trans_matrix = self.math.mat_translate(self.position)
        return self.math.mat_multiply(self.math.mat_multiply(trans_matrix, rot_matrix), scale_matrix)

    def _compute_world_matrix(self):
        """Compute world transformation matrix with parent hierarchy."""
        self.local_matrix = self._compute_local_matrix()
        if self.parent:
            return self.math.mat_multiply(self.parent.world_matrix, self.local_matrix)
        return self.local_matrix

    def set_position(self, position):
        """Set position and update matrices."""
        self.position = self.math.vec3(*position)
        self.world_matrix = self._compute_world_matrix()

    def set_rotation(self, angles):
        """Set Euler rotation (radians) and update matrices."""
        axis = self.math.vec_normalize(self.math.vec3(1, 0, 0))
        self.rotation = self.math.quat(self.math.deg_to_rad(sum(angles) / 3), axis)
        self.world_matrix = self._compute_world_matrix()

    def set_scale(self, scale):
        """Set scale and update matrices."""
        self.scale = scale if isinstance(scale, (int, float)) else self.math.vec3(*scale)
        self.world_matrix = self._compute_world_matrix()

    def get_forward(self):
        """Get forward direction vector."""
        rot_matrix = self.math.quat_to_matrix(self.rotation)
        return self.math.vec_normalize(self.math.mat_multiply(rot_matrix, self.math.vec3(0, 0, -1)))

class Sprite(Component):
    """Component for 2D sprite rendering."""
    def __init__(self, texture_name="default_sprite", uv_coords=((0, 0), (1, 1)), animation_frames=None):
        super().__init__("Sprite")
        self.math = MathLib()
        self.texture_name = texture_name
        self.texture = FALLBACK_TEXTURE_2D
        self.uv_coords = np.array(uv_coords, dtype=np.float32) if self.math.use_numpy else uv_coords
        self.animation_frames = animation_frames or [(0, 0, 1, 1)]
        self.current_frame = 0
        self.frame_time = 0.1
        self.elapsed_time = 0.0

    def update(self, dt):
        """Update sprite animation."""
        if len(self.animation_frames) > 1:
            self.elapsed_time += dt
            if self.elapsed_time >= self.frame_time:
                self.current_frame = (self.current_frame + 1) % len(self.animation_frames)
                self.uv_coords = np.array(self.animation_frames[self.current_frame], dtype=np.float32) if self.math.use_numpy else self.animation_frames[self.current_frame]
                self.elapsed_time = 0.0

    def set_texture(self, texture_name, texture_data=None):
        """Set sprite texture with fallback."""
        self.texture_name = texture_name
        self.texture = texture_data if texture_data is not None else FALLBACK_TEXTURE_2D

class MeshRenderer(Component):
    """Component for 3D mesh rendering with material support."""
    def __init__(self, model_name="cube.obj", material=None, cast_shadows=True):
        super().__init__("MeshRenderer")
        self.math = MathLib()
        self.model_name = model_name
        self.vertices = FALLBACK_CUBE_VERTICES
        self.indices = FALLBACK_CUBE_INDICES
        self.material = material or {
            "diffuse": FALLBACK_TEXTURE_3D_DIFFUSE,
            "normal": FALLBACK_TEXTURE_NORMAL,
            "specular": FALLBACK_TEXTURE_3D_DIFFUSE
        }
        self.cast_shadows = cast_shadows
        self.normals = self._compute_normals()

    def _compute_normals(self):
        """Compute vertex normals for lighting."""
        normals = np.zeros_like(self.vertices) if self.math.use_numpy else [[0, 0, 0] for _ in self.vertices]
        for i in range(0, len(self.indices), 3):
            v0, v1, v2 = [self.vertices[self.indices[i + j]] for j in range(3)]
            edge1 = self.math.vec_sub(v1, v0)
            edge2 = self.math.vec_sub(v2, v0)
            normal = self.math.vec_normalize(self.math.vec_cross(edge1, edge2))
            for j in range(3):
                normals[self.indices[i + j]] = self.math.vec_add(normals[self.indices[i + j]], normal)
        return [self.math.vec_normalize(n) for n in normals] if not self.math.use_numpy else self.math.vec_normalize(normals)

    def set_model(self, model_name, vertices=None, indices=None):
        """Set mesh data with fallback."""
        self.model_name = model_name
        self.vertices = vertices if vertices is not None else FALLBACK_CUBE_VERTICES
        self.indices = indices if indices is not None else FALLBACK_CUBE_INDICES
        self.normals = self._compute_normals()

class Camera(Component):
    """Component for camera properties and projections."""
    def __init__(self, fov=60.0, near=0.1, far=1000.0, is_perspective=True):
        super().__init__("Camera")
        self.math = MathLib()
        self.fov = self.math.deg_to_rad(fov)
        self.aspect = float(WINDOW_SIZE[0]) / WINDOW_SIZE[1]
        self.near = near
        self.far = far
        self.is_perspective = is_perspective
        self.projection_matrix = self._compute_projection()
        self.view_matrix = self.math.mat4()

    def _compute_projection(self):
        """Compute projection matrix."""
        if self.is_perspective:
            return self.math.perspective(self.fov, self.aspect, self.near, self.far)
        return self.math.orthographic(-10, 10, -10, 10, self.near, self.far)

    def update_view(self, transform):
        """Update view matrix based on transform."""
        eye = transform.position
        forward = transform.get_forward()
        target = self.math.vec_add(eye, forward)
        up = self.math.vec3(0, 1, 0)
        z_axis = self.math.vec_normalize(self.math.vec_sub(target, eye))
        x_axis = self.math.vec_normalize(self.math.vec_cross(up, z_axis))
        y_axis = self.math.vec_cross(z_axis, x_axis)
        self.view_matrix = self.math.mat4()
        if self.math.use_numpy:
            self.view_matrix[0:3, 0] = x_axis
            self.view_matrix[0:3, 1] = y_axis
            self.view_matrix[0:3, 2] = z_axis
            self.view_matrix[0:3, 3] = -self.math.vec_dot(x_axis, eye), -self.math.vec_dot(y_axis, eye), -self.math.vec_dot(z_axis, eye)
        else:
            for i in range(3):
                self.view_matrix[i][0], self.view_matrix[i][1], self.view_matrix[i][2] = x_axis[i], y_axis[i], z_axis[i]
            self.view_matrix[0][3], self.view_matrix[1][3], self.view_matrix[2][3] = -self.math.vec_dot(x_axis, eye), -self.math.vec_dot(y_axis, eye), -self.math.vec_dot(z_axis, eye)

class TacticalAI(Component):
    """Component for AI behavior with FSM and pathfinding."""
    def __init__(self, role="neutral", pathfinding="a_star", fsm_state="idle"):
        super().__init__("TacticalAI")
        self.math = MathLib()
        self.role = role
        self.pathfinding = pathfinding
        self.fsm_state = AI_FSM_STATES.get(fsm_state, AI_FSM_STATES["idle"])
        self.path = []
        self.target = None
        self.visibility_range = AI_VISIBILITY_RANGE
        self.speed = 5.0

    def set_target(self, target_position):
        """Set AI target for pathfinding."""
        self.target = self.math.vec3(*target_position)

    def update_path(self, start, goal, grid):
        """Update A* pathfinding."""
        if self.pathfinding == "a_star":
            self.path = self._a_star(start, goal, grid)
        return self.path

    def _a_star(self, start, goal, grid):
        """Simple A* pathfinding implementation."""
        open_list = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.math.vec_magnitude(self.math.vec_sub(goal, start))}
        while open_list:
            current_f, current = min(open_list, key=lambda x: x[0])
            open_list.remove((current_f, current))
            if self.math.vec_magnitude(self.math.vec_sub(current, goal)) < self.math.EPSILON:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            for neighbor in self._get_neighbors(current, grid):
                tentative_g = g_score[current] + self.math.vec_magnitude(self.math.vec_sub(neighbor, current))
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.math.vec_magnitude(self.math.vec_sub(neighbor, goal))
                    open_list.append((f_score[neighbor], neighbor))
        return []

    def _get_neighbors(self, pos, grid):
        """Get valid neighbors for A* pathfinding."""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = self.math.vec_add(pos, self.math.vec3(dx, dy, 0))
            if self.math.point_in_bounds(neighbor, grid):
                neighbors.append(neighbor)
        return neighbors

class Physics3D(Component):
    """Component for 3D physics with collision support."""
    def __init__(self, mass=1.0, velocity=(0, 0, 0), collision_layer="environment"):
        super().__init__("Physics3D")
        self.math = MathLib()
        self.mass = mass
        self.velocity = self.math.vec3(*velocity)
        self.collision_layer = COLLISION_LAYERS.get(collision_layer, COLLISION_LAYERS["environment"])
        self.aabb_min = self.math.vec3(-0.5, -0.5, -0.5)
        self.aabb_max = self.math.vec3(0.5, 0.5, 0.5)

    def apply_force(self, force):
        """Apply force to update velocity."""
        acceleration = self.math.vec_scale(force, 1.0 / self.mass)
        self.velocity = self.math.vec_add(self.velocity, acceleration)

    def update_aabb(self, transform):
        """Update AABB based on transform."""
        self.aabb_min = self.math.vec_add(transform.position, self.math.vec3(-0.5, -0.5, -0.5))
        self.aabb_max = self.math.vec_add(transform.position, self.math.vec3(0.5, 0.5, 0.5))

class Script(Component):
    """Component for scriptable behavior with hot-reloading."""
    def __init__(self, script_path="default_script.py"):
        super().__init__("Script")
        self.script_path = script_path
        self.script_data = FALLBACK_PLAYER_SCRIPT if "player" in script_path else FALLBACK_ENEMY_SCRIPT
        self.module = None
        self.reload()

    def reload(self):
        """Reload script module (placeholder)."""
        try:
            # Simulated script loading
            self.module = type("ScriptModule", (), {"update": lambda e, dt, p, a: None})
        except Exception:
            self.module = type("ScriptModule", (), {"update": lambda e, dt, p, a: None})

    def update(self, entity, dt, physics, assets):
        """Execute script update."""
        if self.module and hasattr(self.module, "update"):
            self.module.update(entity, dt, physics, assets)

class Light(Component):
    """Component for light sources (directional/point)."""
    def __init__(self, type="directional", color=(1, 1, 1), intensity=1.0):
        super().__init__("Light")
        self.math = MathLib()
        self.type = type
        self.color = self.math.vec3(*color)
        self.intensity = intensity
        self.direction = self.math.vec3(0, -1, 0) if type == "directional" else None
        self.range = 10.0 if type == "point" else None

    def set_direction(self, direction):
        """Set light direction for directional lights."""
        if self.type == "directional":
            self.direction = self.math.vec_normalize(self.math.vec3(*direction))

    def get_light_space_matrix(self, transform):
        """Compute light space matrix for shadow mapping."""
        if self.type == "directional":
            light_view = self.math.mat4()
            light_view[0:3, 2] = self.math.vec_scale(self.direction, -1)
            light_view[0:3, 3] = transform.position
            light_proj = self.math.orthographic(-10, 10, -10, 10, -10, 20)
            return self.math.mat_multiply(light_proj, light_view)
        return self.math.mat4()