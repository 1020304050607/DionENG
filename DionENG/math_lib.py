import numpy as np
import math
from .constants import QUADTREE_BOUNDS, WINDOW_SIZE

class MathLib:
    """High-performance math library for DionENG engine."""
    def __init__(self, use_numpy=True):
        """Initialize MathLib with NumPy or pure Python mode."""
        self.use_numpy = use_numpy
        self.EPSILON = 1e-6

    # Vector Operations
    def vec2(self, x, y):
        """Create a 2D vector."""
        return np.array([x, y], dtype=np.float32) if self.use_numpy else [x, y]

    def vec3(self, x, y, z):
        """Create a 3D vector."""
        return np.array([x, y, z], dtype=np.float32) if self.use_numpy else [x, y, z]

    def vec_add(self, a, b):
        """Add two vectors."""
        return a + b if self.use_numpy else [a[i] + b[i] for i in range(len(a))]

    def vec_sub(self, a, b):
        """Subtract two vectors."""
        return a - b if self.use_numpy else [a[i] - b[i] for i in range(len(a))]

    def vec_scale(self, v, s):
        """Scale a vector by a scalar."""
        return v * s if self.use_numpy else [x * s for x in v]

    def vec_dot(self, a, b):
        """Compute dot product of two vectors."""
        return np.dot(a, b) if self.use_numpy else sum(a[i] * b[i] for i in range(len(a)))

    def vec_cross(self, a, b):
        """Compute cross product of two 3D vectors."""
        if self.use_numpy:
            return np.cross(a, b)
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ]

    def vec_normalize(self, v):
        """Normalize a vector."""
        if self.use_numpy:
            norm = np.linalg.norm(v)
            return v / norm if norm > self.EPSILON else v
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm > self.EPSILON else v

    def vec_lerp(self, a, b, t):
        """Linearly interpolate between two vectors."""
        t = max(0.0, min(1.0, t))
        return self.vec_add(self.vec_scale(a, 1 - t), self.vec_scale(b, t))

    def vec_magnitude(self, v):
        """Compute vector magnitude."""
        return np.linalg.norm(v) if self.use_numpy else math.sqrt(sum(x * x for x in v))

    # Matrix Operations
    def mat3(self):
        """Create a 3x3 identity matrix."""
        return np.identity(3, dtype=np.float32) if self.use_numpy else [[1 if i == j else 0 for j in range(3)] for i in range(3)]

    def mat4(self):
        """Create a 4x4 identity matrix."""
        return np.identity(4, dtype=np.float32) if self.use_numpy else [[1 if i == j else 0 for j in range(4)] for i in range(4)]

    def mat_multiply(self, a, b):
        """Multiply two matrices."""
        return np.dot(a, b) if self.use_numpy else [
            [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))
        ]

    def mat_inverse(self, m):
        """Compute matrix inverse."""
        if self.use_numpy:
            return np.linalg.inv(m)
        # Pure Python 4x4 inverse (simplified for clarity)
        det = self.mat_determinant(m)
        if abs(det) < self.EPSILON:
            return m
        # Implement full inverse calculation here (omitted for brevity)
        return m  # Placeholder

    def mat_determinant(self, m):
        """Compute matrix determinant (4x4)."""
        if self.use_numpy:
            return np.linalg.det(m)
        # Simplified determinant for 4x4 (placeholder)
        return 1.0

    def mat_translate(self, v):
        """Create a 4x4 translation matrix."""
        m = self.mat4()
        if self.use_numpy:
            m[:3, 3] = v
        else:
            for i in range(3):
                m[i][3] = v[i]
        return m

    def mat_scale(self, s):
        """Create a 4x4 scale matrix."""
        m = self.mat4()
        if self.use_numpy:
            m[[0, 1, 2], [0, 1, 2]] = s
        else:
            for i in range(3):
                m[i][i] = s[i] if isinstance(s, (list, tuple, np.ndarray)) else s
        return m

    def mat_rotate_x(self, angle):
        """Create a 4x4 rotation matrix around X axis (radians)."""
        c, s = math.cos(angle), math.sin(angle)
        m = self.mat4()
        if self.use_numpy:
            m[1:3, 1:3] = np.array([[c, -s], [s, c]], dtype=np.float32)
        else:
            m[1][1], m[1][2], m[2][1], m[2][2] = c, -s, s, c
        return m

    def mat_rotate_y(self, angle):
        """Create a 4x4 rotation matrix around Y axis (radians)."""
        c, s = math.cos(angle), math.sin(angle)
        m = self.mat4()
        if self.use_numpy:
            m[[0, 2], [0, 2]] = np.array([[c, s], [-s, c]], dtype=np.float32)
        else:
            m[0][0], m[0][2], m[2][0], m[2][2] = c, s, -s, c
        return m

    def mat_rotate_z(self, angle):
        """Create a 4x4 rotation matrix around Z axis (radians)."""
        c, s = math.cos(angle), math.sin(angle)
        m = self.mat4()
        if self.use_numpy:
            m[0:2, 0:2] = np.array([[c, -s], [s, c]], dtype=np.float32)
        else:
            m[0][0], m[0][1], m[1][0], m[1][1] = c, -s, s, c
        return m

    def mat_rotation(self, angles):
        """Create a 4x4 rotation matrix from Euler angles (radians)."""
        rx = self.mat_rotate_x(angles[0])
        ry = self.mat_rotate_y(angles[1])
        rz = self.mat_rotate_z(angles[2])
        return self.mat_multiply(self.mat_multiply(rz, ry), rx)

    # Quaternion Operations
    def quat(self, angle, axis):
        """Create a quaternion from angle (radians) and axis."""
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        axis = self.vec_normalize(axis)
        return np.array([c, s * axis[0], s * axis[1], s * axis[2]], dtype=np.float32) if self.use_numpy else [c, s * axis[0], s * axis[1], s * axis[2]]

    def quat_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ], dtype=np.float32) if self.use_numpy else [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ]

    def quat_to_matrix(self, q):
        """Convert quaternion to 4x4 rotation matrix."""
        w, x, y, z = q
        m = self.mat4()
        if self.use_numpy:
            m[0:3, 0:3] = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
            ], dtype=np.float32)
        else:
            m[0][0] = 1 - 2*y*y - 2*z*z
            m[0][1] = 2*x*y - 2*z*w
            m[0][2] = 2*x*z + 2*y*w
            m[1][0] = 2*x*y + 2*z*w
            m[1][1] = 1 - 2*x*x - 2*z*z
            m[1][2] = 2*y*z - 2*x*w
            m[2][0] = 2*x*z - 2*y*w
            m[2][1] = 2*y*z + 2*x*w
            m[2][2] = 1 - 2*x*x - 2*y*y
        return m

    def quat_slerp(self, q1, q2, t):
        """Spherical linear interpolation between quaternions."""
        t = max(0.0, min(1.0, t))
        cos_theta = self.vec_dot(q1, q2)
        if cos_theta < 0:
            q2 = self.vec_scale(q2, -1)
            cos_theta = -cos_theta
        if cos_theta > 1.0 - self.EPSILON:
            return self.quat_lerp(q1, q2, t)
        theta = math.acos(cos_theta)
        sin_theta = math.sin(theta)
        w1 = math.sin((1 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        return self.vec_add(self.vec_scale(q1, w1), self.vec_scale(q2, w2))

    def quat_lerp(self, q1, q2, t):
        """Linear interpolation between quaternions."""
        t = max(0.0, min(1.0, t))
        return self.vec_normalize(self.vec_add(self.vec_scale(q1, 1 - t), self.vec_scale(q2, t)))

    # Projection Matrices
    def perspective(self, fov, aspect, near, far):
        """Create a perspective projection matrix."""
        f = 1.0 / math.tan(fov / 2)
        m = self.mat4()
        if self.use_numpy:
            m[0, 0] = f / aspect
            m[1, 1] = f
            m[2, 2] = (far + near) / (near - far)
            m[2, 3] = (2 * far * near) / (near - far)
            m[3, 2] = -1
            m[3, 3] = 0
        else:
            m[0][0] = f / aspect
            m[1][1] = f
            m[2][2] = (far + near) / (near - far)
            m[2][3] = (2 * far * near) / (near - far)
            m[3][2] = -1
            m[3][3] = 0
        return m

    def orthographic(self, left, right, bottom, top, near, far):
        """Create an orthographic projection matrix."""
        m = self.mat4()
        if self.use_numpy:
            m[0, 0] = 2 / (right - left)
            m[1, 1] = 2 / (top - bottom)
            m[2, 2] = -2 / (far - near)
            m[0, 3] = -(right + left) / (right - left)
            m[1, 3] = -(top + bottom) / (top - bottom)
            m[2, 3] = -(far + near) / (far - near)
        else:
            m[0][0] = 2 / (right - left)
            m[1][1] = 2 / (top - bottom)
            m[2][2] = -2 / (far - near)
            m[0][3] = -(right + left) / (right - left)
            m[1][3] = -(top + bottom) / (top - bottom)
            m[2][3] = -(far + near) / (far - near)
        return m

    # Geometric Operations
    def ray_triangle_intersection(self, ray_origin, ray_dir, v0, v1, v2):
        """Compute ray-triangle intersection for ray tracing."""
        edge1 = self.vec_sub(v1, v0)
        edge2 = self.vec_sub(v2, v0)
        h = self.vec_cross(ray_dir, edge2)
        a = self.vec_dot(edge1, h)
        if abs(a) < self.EPSILON:
            return None
        f = 1.0 / a
        s = self.vec_sub(ray_origin, v0)
        u = f * self.vec_dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = self.vec_cross(s, edge1)
        v = f * self.vec_dot(ray_dir, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * self.vec_dot(edge2, q)
        if t > self.EPSILON:
            return self.vec_add(ray_origin, self.vec_scale(ray_dir, t))
        return None

    def aabb_intersection(self, a_min, a_max, b_min, b_max):
        """Check if two AABBs intersect."""
        return all(a_max[i] >= b_min[i] and b_max[i] >= a_min[i] for i in range(3))

    def sphere_intersection(self, center1, radius1, center2, radius2):
        """Check if two spheres intersect."""
        distance = self.vec_magnitude(self.vec_sub(center1, center2))
        return distance <= (radius1 + radius2)

    def frustum_cull(self, aabb_min, aabb_max, planes):
        """Check if AABB is inside frustum (6 planes)."""
        for plane in planes:
            p = aabb_max if plane[0] > 0 else aabb_min
            q = aabb_min if plane[0] > 0 else aabb_max
            if self.vec_dot(plane[:3], p) + plane[3] < 0:
                return False
        return True

    # Quadtree Queries
    def point_in_bounds(self, point, bounds):
        """Check if a point is within bounds."""
        x, y = point[:2]
        bx, by, bw, bh = bounds
        return bx <= x < bx + bw and by <= y < by + bh

    def bounds_intersect(self, b1, b2):
        """Check if two bounds intersect."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    # Utility Functions
    def deg_to_rad(self, degrees):
        """Convert degrees to radians."""
        return math.radians(degrees)

    def rad_to_deg(self, radians):
        """Convert radians to degrees."""
        return math.degrees(radians)

    def clamp(self, value, min_val, max_val):
        """Clamp a value between min and max."""
        return max(min_val, min(max_val, value))

    def random_vec3(self, min_val, max_val):
        """Generate a random 3D vector."""
        if self.use_numpy:
            return np.random.uniform(min_val, max_val, 3).astype(np.float32)
        return [np.random.uniform(min_val, max_val) for _ in range(3)]