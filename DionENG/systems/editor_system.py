import pygame
import moderngl
import numpy as np
import msgpack
import json
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock
from collections import defaultdict
from .. import Component, Entity, Transform, MeshRenderer, Camera, TacticalAI, Physics3D, Script

class EditorGizmo:
    """Gizmo for 3D manipulation (translate, rotate, scale)."""
    def __init__(self, mode="translate"):
        self.mode = mode  # translate, rotate, scale
        self.position = np.array([0, 0, 0], dtype='f4')
        self.target_entity = None
        self.axis = None  # x, y, z
        self.active = False

    def update(self, mouse_pos, camera, screen_size):
        """Update gizmo state based on mouse input."""
        if not self.target_entity or not self.active:
            return
        transform = self.target_entity.get_component("Transform")
        if not transform:
            return
        if self.mode == "translate":
            delta = self.screen_to_world_delta(mouse_pos, camera, screen_size)
            if self.axis == "x":
                transform.position[0] += delta[0]
            elif self.axis == "y":
                transform.position[1] += delta[1]
            elif self.axis == "z":
                transform.position[2] += delta[2]
        elif self.mode == "rotate":
            delta = np.array(mouse_pos, dtype='f4') / np.array(screen_size, dtype='f4') * 360
            if self.axis == "x":
                transform.rotation[0] += delta[0]
            elif self.axis == "y":
                transform.rotation[1] += delta[1]
            elif self.axis == "z":
                transform.rotation[2] += delta[2]
        elif self.mode == "scale":
            delta = np.array(mouse_pos, dtype='f4') / np.array(screen_size, dtype='f4')
            transform.scale += delta[0] if self.axis == "all" else 0

    def screen_to_world_delta(self, mouse_pos, camera, screen_size):
        """Convert screen-space mouse delta to world-space."""
        view = np.array(camera.view_matrix, dtype='f4')
        proj = np.array(camera.projection, dtype='f4')
        inv_vp = np.linalg.inv(proj @ view)
        mouse_ndc = np.array([(mouse_pos[0] / screen_size[0]) * 2 - 1, (1 - mouse_pos[1] / screen_size[1]) * 2 - 1, 0, 1], dtype='f4')
        world_pos = inv_vp @ mouse_ndc
        return world_pos[:3] / world_pos[3]

class EditorHandler(FileSystemEventHandler):
    """Handle scene and asset file changes for hot-reloading."""
    def __init__(self, editor_system):
        self.editor_system = editor_system

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            self.editor_system.reload_scene(event.src_path)

class EditorSystem(Component):
    def __init__(self, scene_dir="scenes", screen_size=(1280, 720)):
        super().__init__()
        self.scene_dir = scene_dir
        self.screen_size = np.array(screen_size, dtype='i4')
        self.entities = []
        self.selected_entity = None
        self.gizmo = EditorGizmo()
        self.mode = "select"  # select, place, edit
        self.component_editors = {}
        self.event_queue = []
        self.lock = Lock()
        self.observer = None
        self.renderer2d = None
        self.renderer3d = None
        self.asset_manager = None
        self.memory_system = None
        self.script_system = None
        self.visibility_system = None
        self.ui_system = None
        self.physics_system = None
        self.ctx = None
        self.performance_metrics = defaultdict(float)
        self.grid_size = 1.0
        self.snap_enabled = True
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
        print(f"Editor system initialized: scene_dir={scene_dir}, screen_size={screen_size}")

    def set_renderers(self, renderer2d, renderer3d, ctx):
        """Set 2D and 3D renderers and ModernGL context."""
        self.renderer2d = renderer2d
        self.renderer3d = renderer3d
        self.ctx = ctx
        print("Set 2D/3D renderers and context for editor system")

    def set_asset_manager(self, asset_manager):
        """Set the asset manager for textures and models."""
        self.asset_manager = asset_manager
        print("Set asset manager for editor system")

    def set_memory_system(self, memory_system):
        """Set the memory system for entity allocation."""
        self.memory_system = memory_system
        print("Set memory system for editor system")

    def set_script_system(self, script_system):
        """Set the script system for editor scripting."""
        self.script_system = script_system
        print("Set script system for editor system")

    def set_visibility_system(self, visibility_system):
        """Set the visibility system for 3D culling."""
        self.visibility_system = visibility_system
        print("Set visibility system for editor system")

    def set_ui_system(self, ui_system):
        """Set the UI system for editor interface."""
        self.ui_system = ui_system
        self.setup_ui()
        print("Set UI system for editor system")

    def set_physics_system(self, physics_system):
        """Set the physics system for collider editing."""
        self.physics_system = physics_system
        print("Set physics system for editor system")

    def setup_ui(self):
        """Setup editor UI with buttons and labels."""
        self.ui_system.add_widget("button", "select_mode", [10, 10], [100, 40], text="Select", callback=lambda: self.set_mode("select"))
        self.ui_system.add_widget("button", "place_mode", [120, 10], [100, 40], text="Place", callback=lambda: self.set_mode("place"))
        self.ui_system.add_widget("button", "edit_mode", [230, 10], [100, 40], text="Edit", callback=lambda: self.set_mode("edit"))
        self.ui_system.add_widget("label", "entity_info", [10, 60], [300, 30], text="Selected: None", font_name="arial", font_size=24)
        self.ui_system.add_widget("button", "add_entity", [340, 10], [100, 40], text="Add Entity", callback=self.add_entity)
        self.ui_system.add_widget("button", "save_scene", [450, 10], [100, 40], text="Save Scene", callback=lambda: self.save_scene("scenes/scene.json"))
        print("Setup editor UI")

    def set_mode(self, mode):
        """Set editor mode (select, place, edit)."""
        self.mode = mode
        self.gizmo.active = (mode == "edit")
        print(f"Editor mode set to: {mode}")

    def add_entity(self):
        """Add a new entity to the scene."""
        with self.lock:
            entity = self.memory_system.allocate_entity(f"entity_{len(self.entities)}")
            entity.add_component(self.memory_system.allocate_component("Transform", position=[0, 0, 0]))
            self.entities.append(entity)
            self.selected_entity = entity
            self.ui_system.widgets["entity_info"].text = f"Selected: {entity.name}"
            print(f"Added entity: {entity.name}")

    def select_entity(self, mouse_pos, camera):
        """Select an entity by clicking in 3D space."""
        if not self.visibility_system:
            return
        visible_entities = self.visibility_system.get_visible_entities()
        for entity in visible_entities:
            transform = entity.get_component("Transform")
            if not transform:
                continue
            screen_pos = self.world_to_screen(transform.position, camera)
            if np.linalg.norm(np.array(mouse_pos) - screen_pos) < 20:
                self.selected_entity = entity
                self.ui_system.widgets["entity_info"].text = f"Selected: {entity.name}"
                self.gizmo.target_entity = entity
                print(f"Selected entity: {entity.name}")
                return

    def world_to_screen(self, world_pos, camera):
        """Convert world position to screen coordinates."""
        view = np.array(camera.view_matrix, dtype='f4')
        proj = np.array(camera.projection, dtype='f4')
        pos = np.array([world_pos[0], world_pos[1], world_pos[2], 1], dtype='f4')
        clip = proj @ view @ pos
        ndc = clip[:3] / clip[3]
        return (ndc[:2] + 1) * 0.5 * self.screen_size

    def place_entity(self, mouse_pos, camera):
        """Place an entity at the mouse position with snapping."""
        world_pos = self.screen_to_world(mouse_pos, camera)
        if self.snap_enabled:
            world_pos = np.round(world_pos / self.grid_size) * self.grid_size
        entity = self.memory_system.allocate_entity(f"entity_{len(self.entities)}")
        entity.add_component(self.memory_system.allocate_component("Transform", position=world_pos.tolist()))
        self.entities.append(entity)
        self.selected_entity = entity
        self.ui_system.widgets["entity_info"].text = f"Selected: {entity.name}"
        print(f"Placed entity: {entity.name} at {world_pos}")

    def screen_to_world(self, mouse_pos, camera):
        """Convert screen position to world coordinates."""
        view = np.array(camera.view_matrix, dtype='f4')
        proj = np.array(camera.projection, dtype='f4')
        inv_vp = np.linalg.inv(proj @ view)
        mouse_ndc = np.array([(mouse_pos[0] / self.screen_size[0]) * 2 - 1, (1 - mouse_pos[1] / self.screen_size[1]) * 2 - 1, 0, 1], dtype='f4')
        world_pos = inv_vp @ mouse_ndc
        return world_pos[:3] / world_pos[3]

    def edit_component(self, entity, component_type, property_name, value):
        """Edit a component property."""
        component = entity.get_component(component_type)
        if component:
            setattr(component, property_name, value)
            print(f"Edited {component_type}.{property_name} for {entity.name}")

    def save_scene(self, path):
        """Save the current scene to a JSON file."""
        with self.lock:
            scene = {
                "entities": [
                    {
                        "name": e.name,
                        "components": {
                            c_type: {k: v.tolist() if isinstance(v, np.ndarray) else v
                                     for k, v in c.__dict__.items() if not k.startswith('_')}
                            for c_type, c in e.components.items()
                        }
                    } for e in self.entities
                ]
            }
            with open(path, 'w') as f:
                json.dump(scene, f, indent=2)
            print(f"Saved scene to {path}")

    def load_scene(self, path):
        """Load a scene from a JSON file."""
        with self.lock:
            try:
                with open(path, 'r') as f:
                    scene = json.load(f)
                self.entities.clear()
                for entity_data in scene.get("entities", []):
                    entity = self.memory_system.allocate_entity(entity_data["name"])
                    for c_type, props in entity_data["components"].items():
                        component = self.memory_system.allocate_component(c_type)
                        for k, v in props.items():
                            setattr(component, k, np.array(v) if isinstance(v, list) else v)
                        entity.add_component(component)
                    self.entities.append(entity)
                print(f"Loaded scene: {path}")
            except Exception as e:
                print(f"Error loading scene {path}: {e}")

    async def load_all_scenes(self):
        """Load all scenes in the scene directory."""
        for root, _, files in os.walk(self.scene_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    self.load_scene(path)
        print(f"Loaded {len(self.entities)} entities from scenes")

    def reload_scene(self, path):
        """Reload a modified scene."""
        with self.lock:
            self.entities.clear()
            self.load_scene(path)
            print(f"Hot-reloaded scene: {path}")

    def start_hot_reloading(self):
        """Start monitoring scene directory for changes."""
        if not self.observer:
            self.observer = Observer()
            self.observer.schedule(EditorHandler(self), self.scene_dir, recursive=True)
            self.observer.start()
            print("Started scene hot-reloading")

    def stop_hot_reloading(self):
        """Stop hot-reloading."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            print("Stopped scene hot-reloading")

    def handle_input(self, event, camera):
        """Handle editor input events."""
        with self.lock:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.mode == "select":
                    self.select_entity(event.pos, camera)
                    self.event_queue.append(("select", self.selected_entity.name if self.selected_entity else None))
                elif self.mode == "place":
                    self.place_entity(event.pos, camera)
                    self.event_queue.append(("place", self.selected_entity.name))
                elif self.mode == "edit" and self.gizmo.target_entity:
                    self.gizmo.active = True
                    self.gizmo.axis = "x" if event.button == 1 else "y" if event.button == 2 else "z"
                    self.event_queue.append(("edit", self.gizmo.axis))
            elif event.type == pygame.MOUSEBUTTONUP:
                self.gizmo.active = False
            elif event.type == pygame.MOUSEMOTION and self.gizmo.active:
                self.gizmo.update(event.pos, camera, self.screen_size)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    self.snap_enabled = not self.snap_enabled
                    print(f"Snap enabled: {self.snap_enabled}")
                self.event_queue.append(("key", event.key))
            print(f"Handled editor event: {event.type}, queue_size={len(self.event_queue)}")

    def update(self, entities, camera, dt):
        """Update editor state and script integration."""
        with self.lock:
            self.entities = entities
            if self.script_system:
                for entity in entities:
                    script = entity.get_component("Script")
                    if script and "editor_update" in script.state:
                        script.state["editor_update"](self.entities, self.ui_system.widgets)
            self.performance_metrics["update_time"] += dt
            print(f"Editor update: {len(self.entities)} entities, mode={self.mode}")

    def render_2d(self):
        """Render 2D editor elements (e.g., grid)."""
        if not self.renderer2d:
            return
        with self.lock:
            for x in np.arange(-10, 11, self.grid_size):
                self.renderer2d.add_ui_element(
                    "line", position=[x * 50, -500], end_position=[x * 50, 500], color=(100, 100, 100)
                )
            for y in np.arange(-10, 11, self.grid_size):
                self.renderer2d.add_ui_element(
                    "line", position=[-500, y * 50], end_position=[500, y * 50], color=(100, 100, 100)
                )
            print("Rendered 2D editor elements")

    def render_3d(self, camera):
        """Render 3D editor elements (e.g., gizmos, bounds)."""
        if not self.renderer3d or not self.ctx:
            return
        with self.lock:
            for entity in self.entities:
                transform = entity.get_component("Transform")
                if not transform:
                    continue
                self.renderer3d.add_particle_system(
                    position=transform.position.tolist(),
                    count=1,
                    texture_name="debug_point.png",
                    lifetime=0.1,
                    velocity=(0, 0, 0)
                )
                if entity == self.selected_entity:
                    pos = transform.position
                    size = np.array([1, 1, 1], dtype='f4')
                    vertices = np.array([
                        pos + [-size[0], -size[1], -size[2]], pos + [size[0], -size[1], -size[2]],
                        pos + [size[0], size[1], -size[2]], pos + [-size[0], size[1], -size[2]],
                        pos + [-size[0], -size[1], size[2]], pos + [size[0], -size[1], size[2]],
                        pos + [size[0], size[1], size[2]], pos + [-size[0], size[1], size[2]]
                    ], dtype='f4')
                    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7], dtype='i4')
                    vbo = self.ctx.buffer(vertices.tobytes())
                    ibo = self.ctx.buffer(indices.tobytes())
                    shader = self.ctx.program(
                        vertex_shader='''
                            #version 330
                            in vec3 in_position;
                            uniform mat4 mvp;
                            void main() {
                                gl_Position = mvp * vec4(in_position, 1.0);
                            }
                        ''',
                        fragment_shader='''
                            #version 330
                            out vec4 fragColor;
                            void main() {
                                fragColor = vec4(1.0, 1.0, 0.0, 1.0);
                            }
                        '''
                    )
                    vao = self.ctx.vertex_array(shader, [(vbo, '3f', 'in_position')], ibo)
                    self.renderer3d.add_ui_element(
                        "3d_bounds", vao=vao, mvp=np.array(camera.projection, dtype='f4') @ np.array(camera.view_matrix, dtype='f4')
                    )
            if self.gizmo.active and self.gizmo.target_entity:
                pos = self.gizmo.target_entity.get_component("Transform").position
                self.renderer3d.add_particle_system(
                    position=pos.tolist(),
                    count=1,
                    texture_name="gizmo_point.png",
                    lifetime=0.1,
                    velocity=(0, 0, 0)
                )
            print(f"Rendered 3D editor elements: {len(self.entities)} entities")

    def serialize_scene_state(self):
        """Serialize scene state for network transmission."""
        with self.lock:
            state = {
                "entities": [e.name for e in self.entities],
                "selected": self.selected_entity.name if self.selected_entity else None,
                "mode": self.mode
            }
            return msgpack.packb(state)

    def render_debug(self, renderer2d, renderer3d):
        """Render debug visuals for editor."""
        with self.lock:
            renderer2d.add_ui_element(
                "text", position=[10, 150, 0], content=f"Editor Mode: {self.mode}",
                size=24, color=(255, 255, 255)
            )
            renderer2d.add_ui_element(
                "text", position=[10, 170, 0], content=f"Entities: {len(self.entities)}",
                size=24, color=(255, 255, 255)
            )
            renderer2d.add_ui_element(
                "text", position=[10, 190, 0], content=f"Update: {self.performance_metrics['update_time']:.2f}s",
                size=24, color=(255, 255, 255)
            )
            print("Rendered editor debug info")