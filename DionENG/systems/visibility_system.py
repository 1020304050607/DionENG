import numpy as np
import msgpack
from array import array
from collections import defaultdict
from .. import Component, Entity, Transform, MeshRenderer, Camera, TacticalAI

class AABB:
    """Axis-Aligned Bounding Box for visibility tests."""
    def __init__(self, min_point, max_point):
        self.min_point = np.array(min_point, dtype='f4')
        self.max_point = np.array(max_point, dtype='f4')

    def intersects(self, other):
        """Check if two AABBs intersect."""
        return not (
            self.max_point[0] < other.min_point[0] or
            self.min_point[0] > other.max_point[0] or
            self.max_point[1] < other.min_point[1] or
            self.min_point[1] > other.max_point[1] or
            self.max_point[2] < other.min_point[2] or
            self.min_point[2] > other.max_point[2]
        )

class OctreeNode:
    """Octree node for spatial partitioning."""
    def __init__(self, center, size):
        self.center = np.array(center, dtype='f4')
        self.size = size
        self.entities = []
        self.children = None
        self.aabb = AABB(
            self.center - size / 2,
            self.center + size / 2
        )

    def subdivide(self):
        """Subdivide the node into eight children."""
        half_size = self.size / 2
        quarter_size = self.size / 4
        self.children = [
            OctreeNode(self.center + np.array([x, y, z]) * quarter_size, half_size)
            for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]
        ]
        old_entities = self.entities
        self.entities = []
        for entity in old_entities:
            self.insert(entity)

    def insert(self, entity):
        """Insert an entity into the octree."""
        transform = entity.get_component("Transform")
        if not transform:
            return
        pos = transform.position
        if (self.aabb.min_point <= pos).all() and (pos <= self.aabb.max_point).all():
            if self.children:
                for child in self.children:
                    child.insert(entity)
            else:
                if len(self.entities) > 8 and self.size > 1:
                    self.subdivide()
                    for e in self.entities:
                        self.insert(e)
                    self.entities = []
                self.entities.append(entity)

    def query_frustum(self, frustum):
        """Query entities in the frustum."""
        visible = []
        if not frustum.intersects_aabb(self.aabb):
            return visible
        visible.extend(self.entities)
        if self.children:
            for child in self.children:
                visible.extend(child.query_frustum(frustum))
        return visible

class Frustum:
    """Camera frustum for culling."""
    def __init__(self, camera):
        self.planes = self.compute_planes(camera)

    def compute_planes(self, camera):
        """Compute frustum planes from camera."""
        transform = camera.get_component("Transform")
        projection = np.array(camera.projection, dtype='f4')
        view = np.array(camera.view_matrix, dtype='f4')
        mvp = projection @ view
        planes = []
        for i in range(6):
            row = mvp[:, i // 2] if i % 2 == 0 else -mvp[:, i // 2]
            planes.append(row / np.linalg.norm(row[:3]))
        return planes

    def intersects_aabb(self, aabb):
        """Check if AABB is inside frustum."""
        for plane in self.planes:
            p = aabb.max_point if plane[0] > 0 else aabb.min_point
            p = np.array([p[0], p[1], p[2], 1], dtype='f4')
            if np.dot(plane, p) < 0:
                return False
        return True

class VisibilitySystem(Component):
    def __init__(self, world_size=1000):
        super().__init__()
        self.octree = OctreeNode([0, 0, 0], world_size)
        self.visible_entities = []
        self.lod_levels = {0: 50, 1: 100, 2: 200}  # Distance thresholds for LOD
        self.occluders = []
        self.visibility_cache = {}
        self.debug_data = defaultdict(list)
        print(f"Visibility system initialized: world_size={world_size}")

    def update_octree(self, entities):
        """Update octree with current entities."""
        self.octree = OctreeNode([0, 0, 0], self.octree.size)
        for entity in entities:
            if entity.get_component("Transform") and entity.get_component("MeshRenderer"):
                self.octree.insert(entity)
        print(f"Updated octree with {len(entities)} entities")

    def compute_frustum_culling(self, camera):
        """Perform frustum culling."""
        frustum = Frustum(camera)
        self.visible_entities = self.octree.query_frustum(frustum)
        self.debug_data["frustum"] = [p.tolist() for p in frustum.planes]
        print(f"Frustum culling: {len(self.visible_entities)} entities visible")

    def compute_occlusion_culling(self):
        """Perform occlusion culling using occluders."""
        visible = []
        for entity in self.visible_entities:
            if not self.is_occluded(entity):
                visible.append(entity)
        self.visible_entities = visible
        print(f"Occlusion culling: {len(self.visible_entities)} entities visible")

    def is_occluded(self, entity):
        """Check if entity is occluded by occluders."""
        transform = entity.get_component("Transform")
        if not transform:
            return True
        pos = transform.position
        for occluder in self.occluders:
            occ_transform = occluder.get_component("Transform")
            if not occ_transform:
                continue
            occ_pos = occ_transform.position
            dist = np.linalg.norm(pos - occ_pos)
            if dist < 10:  # Simplified occlusion test
                return True
        return False

    def compute_lod(self, camera):
        """Assign LOD levels based on distance from camera."""
        cam_transform = camera.get_component("Transform")
        if not cam_transform:
            return
        cam_pos = cam_transform.position
        for entity in self.visible_entities:
            transform = entity.get_component("MeshRenderer")
            if transform:
                dist = np.linalg.norm(cam_pos - entity.get_component("Transform").position)
                lod = 0
                for level, threshold in self.lod_levels.items():
                    if dist > threshold:
                        lod = level + 1
                transform.lod_level = min(lod, len(self.lod_levels))
        print(f"Computed LOD for {len(self.visible_entities)} entities")

    def line_of_sight(self, entity1, entity2):
        """Check if entity1 can see entity2 (for AI)."""
        t1 = entity1.get_component("Transform")
        t2 = entity2.get_component("Transform")
        if not t1 or not t2:
            return False
        pos1, pos2 = t1.position, t2.position
        direction = pos2 - pos1
        distance = np.linalg.norm(direction)
        if distance > 10:  # Max LOS distance
            return False
        for occluder in self.occluders:
            occ_transform = occluder.get_component("Transform")
            if not occ_transform:
                continue
            occ_pos = occ_transform.position
            if self.ray_aabb_intersect(pos1, direction / distance, occluder):
                return False
        self.debug_data["los"].append((pos1.tolist(), pos2.tolist()))
        return True

    def ray_aabb_intersect(self, origin, direction, entity):
        """Ray-AABB intersection test for LOS."""
        transform = entity.get_component("Transform")
        if not transform:
            return False
        aabb = AABB(
            transform.position - np.array([1, 1, 1]),
            transform.position + np.array([1, 1, 1])
        )
        t_min, t_max = 0, float('inf')
        for i in range(3):
            if abs(direction[i]) < 1e-6:
                if origin[i] < aabb.min_point[i] or origin[i] > aabb.max_point[i]:
                    return False
            else:
                t1 = (aabb.min_point[i] - origin[i]) / direction[i]
                t2 = (aabb.max_point[i] - origin[i]) / direction[i]
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
        return t_min <= t_max

    def update(self, entities, camera, asset_manager, dt):
        """Update visibility system."""
        self.occluders = [e for e in entities if e.get_component("MeshRenderer") and e.get_component("Transform")]
        self.compute_frustum_culling(camera)
        self.compute_occlusion_culling()
        self.compute_lod(camera)
        self.update_octree(entities)
        print(f"Visibility update: {len(self.visible_entities)} entities visible")

    def get_visible_entities(self):
        """Get list of visible entities for rendering."""
        return self.visible_entities

    def serialize_visibility(self):
        """Serialize visibility data for network transmission."""
        visible_ids = [e.name for e in self.visible_entities]
        return msgpack.packb({"visible_entities": visible_ids})

    def render_debug(self, renderer):
        """Render debug visuals for frustum, AABBs, and LOS."""
        for entity in self.visible_entities:
            transform = entity.get_component("Transform")
            if transform:
                aabb = AABB(
                    transform.position - np.array([1, 1, 1]),
                    transform.position + np.array([1, 1, 1])
                )
                renderer.add_particle_system(
                    position=aabb.min_point.tolist(),
                    count=1,
                    texture_name="debug_point.png",
                    lifetime=0.1,
                    velocity=(0, 0, 0)
                )
                renderer.add_particle_system(
                    position=aabb.max_point.tolist(),
                    count=1,
                    texture_name="debug_point.png",
                    lifetime=0.1,
                    velocity=(0, 0, 0)
                )
        for los in self.debug_data["los"]:
            renderer.add_particle_system(
                position=los[0],
                count=1,
                texture_name="debug_point.png",
                lifetime=0.1,
                velocity=(0, 0, 0)
            )
        print("Rendered visibility debug info")