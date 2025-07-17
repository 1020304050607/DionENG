import uuid
import copy
import msgpack
from typing import Dict, Optional, List
from .components import Transform, Sprite, MeshRenderer, Camera, TacticalAI, Physics3D, Script, Light
from .math_lib import MathLib
from .constants import ECS_MAX_ENTITIES
from .fallbacks import FALLBACK_SCENE

class Entity:
    """Entity class for DionENG's ECS architecture."""
    _id_counter = 0  # Class-level counter for unique IDs

    def __init__(self, name: str = "Entity", tags: List[str] = None):
        """Initialize an entity with a unique ID, name, and optional tags."""
        if Entity._id_counter >= ECS_MAX_ENTITIES:
            raise ValueError(f"Cannot create entity: Exceeded ECS_MAX_ENTITIES ({ECS_MAX_ENTITIES})")
        self.id = uuid.uuid4().hex  # Unique identifier
        self.name = name
        self.tags = set(tags or [])  # Set for fast lookup
        self.components: Dict[str, Component] = {}  # Component storage
        self.active = True  # Active state for systems
        self.math = MathLib()
        Entity._id_counter += 1

    def __str__(self) -> str:
        """String representation of the entity."""
        return f"Entity(id={self.id[:8]}, name={self.name}, tags={self.tags}, components={list(self.components.keys())})"

    def add_component(self, component: Component) -> None:
        """Add a component to the entity."""
        if component.type in self.components:
            raise ValueError(f"Component {component.type} already exists on entity {self.name}")
        self.components[component.type] = component
        if isinstance(component, Transform) and self.get_component("Transform") != component:
            raise ValueError("Only one Transform component allowed per entity")
        if isinstance(component, Camera) and self.get_component("Camera") != component:
            raise ValueError("Only one Camera component allowed per entity")

    def remove_component(self, component_type: str) -> None:
        """Remove a component by type."""
        if component_type in self.components:
            del self.components[component_type]

    def get_component(self, component_type: str) -> Optional[Component]:
        """Get a component by type."""
        return self.components.get(component_type)

    def has_component(self, component_type: str) -> bool:
        """Check if entity has a component."""
        return component_type in self.components

    def has_tag(self, tag: str) -> bool:
        """Check if entity has a specific tag."""
        return tag in self.tags

    def add_tag(self, tag: str) -> None:
        """Add a tag to the entity."""
        self.tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the entity."""
        self.tags.discard(tag)

    def serialize(self) -> bytes:
        """Serialize entity state for networking or saving."""
        data = {
            "id": self.id,
            "name": self.name,
            "tags": list(self.tags),
            "active": self.active,
            "components": {}
        }
        for comp_type, comp in self.components.items():
            comp_data = {}
            if isinstance(comp, Transform):
                comp_data = {
                    "position": comp.position.tolist() if comp.math.use_numpy else comp.position,
                    "rotation": comp.rotation.tolist() if comp.math.use_numpy else comp.rotation,
                    "scale": comp.scale.tolist() if isinstance(comp.scale, (np.ndarray, list)) else comp.scale,
                    "parent_id": comp.parent.id if comp.parent else None
                }
            elif isinstance(comp, Sprite):
                comp_data = {
                    "texture_name": comp.texture_name,
                    "uv_coords": comp.uv_coords.tolist() if comp.math.use_numpy else comp.uv_coords,
                    "animation_frames": comp.animation_frames,
                    "current_frame": comp.current_frame,
                    "frame_time": comp.frame_time
                }
            elif isinstance(comp, MeshRenderer):
                comp_data = {
                    "model_name": comp.model_name,
                    "cast_shadows": comp.cast_shadows,
                    "material": {
                        "diffuse": comp.material["diffuse"].tolist() if comp.math.use_numpy else comp.material["diffuse"],
                        "normal": comp.material["normal"].tolist() if comp.math.use_numpy else comp.material["normal"],
                        "specular": comp.material["specular"].tolist() if comp.math.use_numpy else comp.material["specular"]
                    }
                }
            elif isinstance(comp, Camera):
                comp_data = {
                    "fov": comp.fov,
                    "aspect": comp.aspect,
                    "near": comp.near,
                    "far": comp.far,
                    "is_perspective": comp.is_perspective
                }
            elif isinstance(comp, TacticalAI):
                comp_data = {
                    "role": comp.role,
                    "pathfinding": comp.pathfinding,
                    "fsm_state": comp.fsm_state,
                    "target": comp.target.tolist() if comp.target is not None and comp.math.use_numpy else comp.target,
                    "speed": comp.speed
                }
            elif isinstance(comp, Physics3D):
                comp_data = {
                    "mass": comp.mass,
                    "velocity": comp.velocity.tolist() if comp.math.use_numpy else comp.velocity,
                    "collision_layer": comp.collision_layer,
                    "aabb_min": comp.aabb_min.tolist() if comp.math.use_numpy else comp.aabb_min,
                    "aabb_max": comp.aabb_max.tolist() if comp.math.use_numpy else comp.aabb_max
                }
            elif isinstance(comp, Script):
                comp_data = {"script_path": comp.script_path}
            elif isinstance(comp, Light):
                comp_data = {
                    "type": comp.type,
                    "color": comp.color.tolist() if comp.math.use_numpy else comp.color,
                    "intensity": comp.intensity,
                    "direction": comp.direction.tolist() if comp.direction is not None and comp.math.use_numpy else comp.direction,
                    "range": comp.range
                }
            data["components"][comp_type] = comp_data
        return msgpack.packb(data)

    def deserialize(self, data: bytes) -> None:
        """Deserialize entity state from bytes."""
        try:
            deserialized = msgpack.unpackb(data, raw=False)
            self.id = deserialized.get("id", self.id)
            self.name = deserialized.get("name", self.name)
            self.tags = set(deserialized.get("tags", []))
            self.active = deserialized.get("active", True)
            self.components.clear()
            for comp_type, comp_data in deserialized.get("components", {}).items():
                if comp_type == "Transform":
                    parent = None  # Parent will need to be resolved post-deserialization
                    self.add_component(Transform(
                        position=comp_data.get("position", (0, 0, 0)),
                        rotation=comp_data.get("rotation", (0, 0, 0)),
                        scale=comp_data.get("scale", 1.0),
                        parent=parent
                    ))
                elif comp_type == "Sprite":
                    self.add_component(Sprite(
                        texture_name=comp_data.get("texture_name", "default_sprite"),
                        uv_coords=comp_data.get("uv_coords", ((0, 0), (1, 1))),
                        animation_frames=comp_data.get("animation_frames", [(0, 0, 1, 1)])
                    ))
                elif comp_type == "MeshRenderer":
                    self.add_component(MeshRenderer(
                        model_name=comp_data.get("model_name", "cube.obj"),
                        material=comp_data.get("material", {
                            "diffuse": FALLBACK_TEXTURE_3D_DIFFUSE,
                            "normal": FALLBACK_TEXTURE_NORMAL,
                            "specular": FALLBACK_TEXTURE_3D_DIFFUSE
                        }),
                        cast_shadows=comp_data.get("cast_shadows", True)
                    ))
                elif comp_type == "Camera":
                    self.add_component(Camera(
                        fov=comp_data.get("fov", 60.0),
                        near=comp_data.get("near", 0.1),
                        far=comp_data.get("far", 1000.0),
                        is_perspective=comp_data.get("is_perspective", True)
                    ))
                elif comp_type == "TacticalAI":
                    self.add_component(TacticalAI(
                        role=comp_data.get("role", "neutral"),
                        pathfinding=comp_data.get("pathfinding", "a_star"),
                        fsm_state=comp_data.get("fsm_state", "idle")
                    ))
                elif comp_type == "Physics3D":
                    self.add_component(Physics3D(
                        mass=comp_data.get("mass", 1.0),
                        velocity=comp_data.get("velocity", (0, 0, 0)),
                        collision_layer=comp_data.get("collision_layer", "environment")
                    ))
                elif comp_type == "Script":
                    self.add_component(Script(
                        script_path=comp_data.get("script_path", "default_script.py")
                    ))
                elif comp_type == "Light":
                    self.add_component(Light(
                        type=comp_data.get("type", "directional"),
                        color=comp_data.get("color", (1, 1, 1)),
                        intensity=comp_data.get("intensity", 1.0)
                    ))
        except Exception as e:
            print(f"Deserialization failed: {e}")
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Load fallback entity state from FALLBACK_SCENE."""
        fallback_entity = FALLBACK_SCENE["entities"][0]
        self.components.clear()
        for comp_type, comp_data in fallback_entity["components"].items():
            if comp_type == "Transform":
                self.add_component(Transform(**comp_data))
            elif comp_type == "MeshRenderer":
                self.add_component(MeshRenderer(**comp_data))

    def clone(self) -> 'Entity':
        """Create a deep copy of the entity."""
        new_entity = Entity(name=f"{self.name}_clone", tags=self.tags)
        new_entity.active = self.active
        for comp_type, comp in self.components.items():
            new_entity.add_component(copy.deepcopy(comp))
        return new_entity

    def update_transform_hierarchy(self) -> None:
        """Update transform hierarchy for parent-child relationships."""
        transform = self.get_component("Transform")
        if transform:
            transform.world_matrix = transform._compute_world_matrix()

    def get_world_position(self) -> Optional[np.ndarray]:
        """Get world position from Transform component."""
        transform = self.get_component("Transform")
        return transform.position if transform else None

    def get_components_by_types(self, component_types: List[str]) -> List[Component]:
        """Get all components matching the specified types."""
        return [comp for comp_type, comp in self.components.items() if comp_type in component_types]

    def destroy(self) -> None:
        """Mark entity as inactive and clear components."""
        self.active = False
        self.components.clear()
        Entity._id_counter -= 1

    def __del__(self):
        """Clean up entity resources."""
        self.destroy()