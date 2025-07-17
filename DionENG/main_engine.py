import asyncio
import logging
import pygame
import moderngl
import pyopencl as cl
import numpy as np
import msgpack
import comtypes.client
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict, deque
from .systems import (
    InputSystem, AudioSystem, PhysicsSystem2D, PhysicsSystem3D, RenderSystem2D, RenderSystem3D,
    AISystem, NetworkSystem, AssetManager, MemorySystem, VisibilitySystem, ScriptSystem,
    UISystem, EditorSystem, JobSystem
)
from .profiling import Profiler
from .scene import Scene
from .entity import Entity
from .components import Transform, MeshRenderer, Camera, TacticalAI, Physics3D, Script, Light

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dioneng.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DionENG')

class QuadTree:
    """Simple quadtree for 2D spatial partitioning."""
    def __init__(self, bounds, depth=0, max_depth=8, max_objects=10):
        self.bounds = bounds  # [x, y, width, height]
        self.depth = depth
        self.max_depth = max_depth
        self.max_objects = max_objects
        self.objects = []
        self.nodes = []

    def insert(self, entity, position):
        """Insert an entity into the quadtree."""
        if not self.nodes:
            if len(self.objects) < self.max_objects or self.depth >= self.max_depth:
                self.objects.append((entity, position))
                return
            self.subdivide()
        for node in self.nodes:
            if self.point_in_bounds(position, node.bounds):
                node.insert(entity, position)
                return
        self.objects.append((entity, position))

    def subdivide(self):
        """Subdivide the quadtree into four nodes."""
        x, y, w, h = self.bounds
        hw, hh = w / 2, h / 2
        self.nodes = [
            QuadTree([x, y, hw, hh], self.depth + 1, self.max_depth, self.max_objects),
            QuadTree([x + hw, y, hw, hh], self.depth + 1, self.max_depth, self.max_objects),
            QuadTree([x, y + hh, hw, hh], self.depth + 1, self.max_depth, self.max_objects),
            QuadTree([x + hw, y + hh, hw, hh], self.depth + 1, self.max_depth, self.max_objects)
        ]
        for obj, pos in self.objects:
            for node in self.nodes:
                if self.point_in_bounds(pos, node.bounds):
                    node.insert(obj, pos)
        self.objects = []

    def point_in_bounds(self, point, bounds):
        """Check if a point is within bounds."""
        x, y = point[:2]
        bx, by, bw, bh = bounds
        return bx <= x < bx + bw and by <= y < by + bh

    def query(self, bounds):
        """Query entities within bounds."""
        results = []
        if self.nodes:
            for node in self.nodes:
                if self.bounds_intersect(bounds, node.bounds):
                    results.extend(node.query(bounds))
        results.extend(self.objects)
        return results

    def bounds_intersect(self, b1, b2):
        """Check if two bounds intersect."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

class BaseRenderer:
    """Base class for rendering systems with advanced features."""
    def __init__(self, window_size):
        self.window_size = np.array(window_size, dtype='i4')
        self.quality = 1.0
        self.post_effects = {'bloom': False, 'ssao': False, 'motion_blur': False}
        self.shadow_map = None
        self.batch_data = defaultdict(list)

    def set_quality(self, quality):
        """Set rendering quality for dynamic resolution."""
        self.quality = max(0.5, min(2.0, quality))
        logger.info(f"Set render quality: {self.quality}")

    def enable_post_effect(self, effect):
        """Enable a post-processing effect."""
        if effect in self.post_effects:
            self.post_effects[effect] = True
            logger.info(f"Enabled post effect: {effect}")

    def disable_post_effect(self, effect):
        """Disable a post-processing effect."""
        if effect in self.post_effects:
            self.post_effects[effect] = False
            logger.info(f"Disabled post effect: {effect}")

    def batch_geometry(self, entities):
        """Batch entities with same material for rendering."""
        self.batch_data.clear()
        for entity in entities:
            renderer = entity.get_component("MeshRenderer")
            if renderer:
                material = renderer.material or "default"
                self.batch_data[material].append(entity)
        logger.debug(f"Batched {len(self.batch_data)} material groups")

    def render_shadow_map(self, light, entities):
        """Render shadow map for a light."""
        if not self.shadow_map:
            logger.debug("Initializing shadow map")
            # Placeholder for shadow map rendering
        logger.debug(f"Rendered shadow map for light: {light}")

class GameEngine:
    def __init__(self, window_size=(1280, 720), title="DionENG", editor_mode=False):
        """Initialize the game engine with all systems."""
        self.window_size = np.array(window_size, dtype='i4')
        self.title = title
        self.editor_mode = editor_mode
        self.running = False
        self.delta_time = 0.0
        self.fixed_dt = 1.0 / 60.0
        self.target_fps = 60
        self.min_dt = 1.0 / 240.0
        self.clock = pygame.time.Clock()
        self.scene = Scene()
        self.profiler = Profiler()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.entity_pool = deque(maxlen=1000)
        self.active_cameras = [None]
        self.quadtree = QuadTree([0, 0, 1000, 1000])
        self.input_mappings = {
            'move_forward': pygame.K_w,
            'move_backward': pygame.K_s,
            'move_left': pygame.K_a,
            'move_right': pygame.K_d
        }

        # Initialize Pygame and ModernGL
        try:
            pygame.init()
            pygame.display.set_mode(window_size, pygame.OPENGL | pygame.DOUBLEBUF)
            self.ctx = moderngl.create_context()
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
            logger.info("Initialized Pygame and ModernGL")
        except Exception as e:
            logger.error(f"Failed to initialize Pygame/ModernGL: {e}")
            raise

        # Initialize OpenCL for ray tracing and particles
        try:
            self.cl_platform = cl.get_platforms()[0]
            self.cl_device = self.cl_platform.get_devices()[0]
            self.cl_ctx = cl.Context([self.cl_device])
            self.cl_queue = cl.CommandQueue(self.cl_ctx)
            logger.info("Initialized OpenCL")
        except Exception as e:
            logger.warning(f"OpenCL initialization failed: {e}, falling back to CPU")

        # Initialize DirectX (simulated via comtypes)
        try:
            self.dx_device = None  # Placeholder for DirectX simulation
            logger.info("Initialized DirectX simulation")
        except Exception as e:
            logger.warning(f"DirectX initialization failed: {e}")

        # Initialize systems
        try:
            self.memory = MemorySystem(max_entities=10000, max_components=100000)
            self.assets = AssetManager(asset_dir="assets")
            self.assets.set_context(self.ctx)
            self.job_system = JobSystem()
            self.input = InputSystem()
            self.audio = AudioSystem()
            self.physics2d = PhysicsSystem2D()
            self.physics3d = PhysicsSystem3D()
            self.renderer2d = RenderSystem2D(window_size)
            self.renderer3d = RenderSystem3D(self.ctx)
            self.ai = AISystem()
            self.network = NetworkSystem()
            self.visibility = VisibilitySystem(world_size=1000)
            self.script = ScriptSystem(script_dir="scripts")
            self.ui = UISystem(ui_dir="ui", screen_size=window_size)
            self.editor = EditorSystem(scene_dir="scenes", screen_size=window_size)
            logger.info("Initialized all systems")
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

        # Set system dependencies
        try:
            self.ui.set_renderers(self.renderer2d, self.renderer3d, self.ctx)
            self.ui.set_asset_manager(self.assets)
            self.ui.set_script_system(self.script)
            self.ui.set_visibility_system(self.visibility)
            self.editor.set_renderers(self.renderer2d, self.renderer3d, self.ctx)
            self.editor.set_asset_manager(self.assets)
            self.editor.set_memory_system(self.memory)
            self.editor.set_script_system(self.script)
            self.editor.set_visibility_system(self.visibility)
            self.editor.set_ui_system(self.ui)
            self.editor.set_physics_system(self.physics3d)
            self.renderer3d.set_asset_manager(self.assets)
            self.renderer3d.set_visibility_system(self.visibility)
            self.ai.set_physics_system(self.physics3d)
            self.network.set_scene(self.scene)
            self.script.set_physics_system(self.physics3d)
            self.script.set_asset_manager(self.assets)
            logger.info("Set system dependencies")
        except Exception as e:
            logger.error(f"System dependency setup failed: {e}")
            raise

        # Initialize main camera
        try:
            self.main_camera = self.memory.allocate_entity("main_camera")
            self.main_camera.add_component(self.memory.allocate_component("Transform", position=[0, 0, 10]))
            self.main_camera.add_component(self.memory.allocate_component("Camera"))
            self.scene.add_entity(self.main_camera)
            self.active_cameras[0] = self.main_camera
            logger.info("Initialized main camera")
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise

    async def load_initial_content(self):
        """Load initial assets, scripts, UI, and scenes."""
        try:
            await self.assets.load_all_assets()
            await self.script.load_all_scripts(self.scene.entities)
            await self.ui.load_all_ui_layouts()
            await self.editor.load_all_scenes()
            if self.editor_mode:
                self.editor.start_hot_reloading()
                self.script.start_hot_reloading()
                self.ui.start_hot_reloading()
            logger.info("Loaded initial content")
        except Exception as e:
            logger.error(f"Failed to load initial content: {e}")

    def handle_input(self):
        """Handle input events with remapping and buffering."""
        try:
            input_buffer = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_e:
                        self.toggle_editor_mode()
                    elif event.key in self.input_mappings.values():
                        input_buffer.append(('key', event.key))
                elif event.type == pygame.JOYBUTTONDOWN:
                    input_buffer.append(('gamepad', event.button))
                if self.editor_mode:
                    self.editor.handle_input(event, self.main_camera)
                else:
                    self.input.handle_event(event, self.scene.entities)
                    self.ui.handle_input(event, self.main_camera)
            self.input.buffer_inputs(input_buffer)
            self.profiler.record("input_time", self.clock.get_time() / 1000.0)
            logger.debug(f"Processed {len(input_buffer)} input events")
        except Exception as e:
            logger.error(f"Input handling failed: {e}")

    def update_systems(self):
        """Update systems with adaptive timestep and multithreading."""
        try:
            adaptive_dt = min(self.fixed_dt * (self.clock.get_fps() / self.target_fps), self.min_dt)
            self.quadtree = QuadTree([0, 0, 1000, 1000])
            for entity in self.scene.entities:
                transform = entity.get_component("Transform")
                if transform:
                    self.quadtree.insert(entity, transform.position)
            self.job_system.add_job(
                lambda: self.input.update(self.scene.entities, adaptive_dt)
            )
            self.job_system.add_job(
                lambda: self.audio.update(self.scene.entities, adaptive_dt)
            )
            self.job_system.add_job(
                lambda: self.physics2d.update(self.scene.entities, adaptive_dt, self.visibility, self.quadtree)
            )
            self.job_system.add_job(
                lambda: self.physics3d.update(self.scene.entities, adaptive_dt, self.visibility, self.quadtree)
            )
            self.job_system.add_job(
                lambda: self.ai.update(self.scene.entities, self.physics3d, adaptive_dt, self.quadtree)
            )
            self.job_system.add_job(
                lambda: self.network.update(self.scene.entities, adaptive_dt, delta_compression=True)
            )
            self.job_system.add_job(
                lambda: self.script.update(self.scene.entities, self.physics3d, self.assets, adaptive_dt)
            )
            self.job_system.add_job(
                lambda: self.ui.update(self.scene.entities, self.main_camera, adaptive_dt)
            )
            if self.editor_mode:
                self.job_system.add_job(
                    lambda: self.editor.update(self.scene.entities, self.main_camera, adaptive_dt)
                )
            self.executor.submit(self.job_system.run_jobs)
            self.profiler.record("update_time", self.clock.get_time() / 1000.0)
            logger.debug(f"Updated systems: {len(self.job_system.jobs)} jobs")
        except Exception as e:
            logger.error(f"System update failed: {e}")

    def render_2d(self, camera_index=0):
        """Render 2D elements with dynamic resolution."""
        if not self.renderer2d:
            return
        try:
            resolution_scale = self.calculate_dynamic_resolution()
            self.renderer2d.set_resolution(self.window_size * resolution_scale)
            self.renderer2d.clear()
            visible_entities = self.quadtree.query(self.get_camera_bounds(self.active_cameras[camera_index]))
            self.renderer2d.batch_geometry(visible_entities)
            self.renderer2d.update(visible_entities, self.active_cameras[camera_index], self.fixed_dt)
            self.ui.render_2d()
            if self.editor_mode:
                self.editor.render_2d()
            self.renderer2d.present()
            self.profiler.record("render_2d_time", self.clock.get_time() / 1000.0)
            logger.debug("Rendered 2D elements")
        except Exception as e:
            logger.error(f"2D rendering failed: {e}")

    def render_3d(self, camera_index=0):
        """Render 3D elements with ray tracing and shadows."""
        if not self.renderer3d:
            return
        try:
            self.ctx.clear(0.1, 0.1, 0.1, 1.0)
            camera = self.active_cameras[camera_index]
            visible_entities = self.visibility.get_visible_entities(camera)
            self.renderer3d.batch_geometry(visible_entities)
            lights = [e.get_component("Light") for e in self.scene.entities if e.get_component("Light")]
            for light in lights:
                self.renderer3d.render_shadow_map(light, visible_entities)
            if self.renderer3d.post_effects.get('ray_tracing', False):
                self.run_ray_tracing(visible_entities, camera)
            else:
                self.renderer3d.update(visible_entities, camera, self.fixed_dt)
            self.ui.render_3d(camera)
            if self.editor_mode:
                self.editor.render_3d(camera)
            self.profiler.record("render_3d_time", self.clock.get_time() / 1000.0)
            logger.debug(f"Rendered 3D elements: {len(visible_entities)} visible entities")
        except Exception as e:
            logger.error(f"3D rendering failed: {e}")

    def run_ray_tracing(self, entities, camera):
        """Run ray tracing using OpenCL."""
        try:
            kernel = """
            __kernel void ray_trace(__global float* positions, __global float* normals, __global float* output, int width, int height) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                if (x >= width || y >= height) return;
                float3 ray_dir = normalize((float3)(x / (float)width - 0.5, y / (float)height - 0.5, 1.0));
                float3 color = (float3)(0.0, 0.0, 0.0);
                for (int i = 0; i < positions_count; i += 3) {
                    float3 pos = (float3)(positions[i], positions[i+1], positions[i+2]);
                    float t = dot(ray_dir, pos);
                    if (t > 0.0) color += (float3)(0.1, 0.1, 0.1);
                }
                output[y * width + x] = color.x;
            }
            """
            program = cl.Program(self.cl_ctx, kernel).build()
            positions = np.array([e.get_component("Transform").position for e in entities if e.get_component("Transform")], dtype=np.float32)
            normals = np.array([e.get_component("MeshRenderer").normal if e.get_component("MeshRenderer") else [0, 0, 1] for e in entities], dtype=np.float32)
            output = np.zeros(self.window_size, dtype=np.float32)
            pos_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=positions)
            norm_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=normals)
            out_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
            program.ray_trace(self.cl_queue, self.window_size, None, pos_buf, norm_buf, out_buf, np.int32(self.window_size[0]), np.int32(self.window_size[1]))
            cl.enqueue_copy(self.cl_queue, output, out_buf)
            self.renderer3d.apply_ray_tracing_output(output)
            logger.debug("Executed ray tracing")
        except Exception as e:
            logger.error(f"Ray tracing failed: {e}")

    def calculate_dynamic_resolution(self):
        """Calculate dynamic resolution based on performance."""
        try:
            fps = self.clock.get_fps()
            target_fps = self.target_fps
            if fps < target_fps * 0.8:
                return max(0.5, self.renderer2d.quality * 0.9)
            elif fps > target_fps * 1.2:
                return min(2.0, self.renderer2d.quality * 1.1)
            return self.renderer2d.quality
        except Exception as e:
            logger.error(f"Dynamic resolution calculation failed: {e}")
            return 1.0

    def get_camera_bounds(self, camera):
        """Get 2D bounds for a camera's frustum."""
        transform = camera.get_component("Transform")
        cam = camera.get_component("Camera")
        if transform and cam:
            return [transform.position[0] - cam.fov, transform.position[1] - cam.fov, cam.fov * 2, cam.fov * 2]
        return [0, 0, 1000, 1000]

    def render_debug(self):
        """Render debug information with frame debugger."""
        try:
            if self.editor_mode:
                self.editor.render_debug(self.renderer2d, self.renderer3d)
            self.ui.render_debug(self.renderer2d, self.renderer3d)
            self.script.render_debug(self.renderer3d)
            self.profiler.render(self.renderer2d, {
                'entities': len(self.scene.entities),
                'visible': len(self.visibility.get_visible_entities(self.main_camera)),
                'jobs': len(self.job_system.jobs)
            })
            self.profiler.record("debug_time", self.clock.get_time() / 1000.0)
            logger.debug("Rendered debug info")
        except Exception as e:
            logger.error(f"Debug rendering failed: {e}")

    async def run(self):
        """Main game loop with adaptive timestep."""
        self.running = True
        try:
            await self.load_initial_content()
            last_time = pygame.time.get_ticks() / 1000.0
            lag = 0.0

            while self.running:
                current_time = pygame.time.get_ticks() / 1000.0
                self.delta_time = current_time - last_time
                last_time = current_time
                lag += self.delta_time

                self.handle_input()

                while lag >= self.fixed_dt:
                    self.update_systems()
                    lag -= self.fixed_dt

                for i, camera in enumerate(self.active_cameras):
                    if camera:
                        self.render_2d(i)
                        self.render_3d(i)
                self.render_debug()

                pygame.display.flip()
                self.clock.tick(self.target_fps)
                self.profiler.record("fps", self.clock.get_fps())

                await asyncio.sleep(0)

            self.cleanup()
            logger.info("Game loop terminated")
        except Exception as e:
            logger.error(f"Game loop failed: {e}")
            self.cleanup()

    def cleanup(self):
        """Cleanup resources and shutdown systems."""
        try:
            self.script.stop_hot_reloading()
            self.ui.stop_hot_reloading()
            self.editor.stop_hot_reloading()
            self.job_system.shutdown()
            self.audio.shutdown()
            self.executor.shutdown()
            pygame.quit()
            logger.info("Engine cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def add_entity(self, name, components=None, parent=None):
        """Add an entity with archetype-based storage."""
        try:
            entity = self.entity_pool.pop() if self.entity_pool else self.memory.allocate_entity(name)
            if components:
                for comp_type, props in components.items():
                    component = self.memory.allocate_component(comp_type, **props)
                    entity.add_component(component)
            if parent:
                parent_transform = parent.get_component("Transform")
                entity_transform = entity.get_component("Transform")
                if parent_transform and entity_transform:
                    entity_transform.parent = parent_transform
            self.scene.add_entity(entity)
            transform = entity.get_component("Transform")
            if transform:
                self.quadtree.insert(entity, transform.position)
            logger.info(f"Added entity: {name}")
            return entity
        except Exception as e:
            logger.error(f"Failed to add entity {name}: {e}")
            return None

    def remove_entity(self, name):
        """Remove an entity and return to pool."""
        try:
            entity = next((e for e in self.scene.entities if e.name == name), None)
            if entity:
                self.scene.remove_entity(entity)
                self.entity_pool.append(entity)
                logger.info(f"Removed entity: {name}")
        except Exception as e:
            logger.error(f"Failed to remove entity {name}: {e}")

    def set_camera_position(self, camera_index, position):
        """Set camera position."""
        try:
            if camera_index < len(self.active_cameras):
                transform = self.active_cameras[camera_index].get_component("Transform")
                if transform:
                    transform.position = np.array(position, dtype='f4')
                    logger.info(f"Set camera {camera_index} position: {position}")
        except Exception as e:
            logger.error(f"Failed to set camera position: {e}")

    def add_camera(self):
        """Add a new camera for multi-camera rendering."""
        try:
            camera = self.memory.allocate_entity(f"camera_{len(self.active_cameras)}")
            camera.add_component(self.memory.allocate_component("Transform", position=[0, 0, 10]))
            camera.add_component(self.memory.allocate_component("Camera"))
            self.scene.add_entity(camera)
            self.active_cameras.append(camera)
            logger.info(f"Added camera: {camera.name}")
            return len(self.active_cameras) - 1
        except Exception as e:
            logger.error(f"Failed to add camera: {e}")
            return None

    def toggle_editor_mode(self):
        """Toggle between runtime and editor modes."""
        try:
            self.editor_mode = not self.editor_mode
            if self.editor_mode:
                self.editor.start_hot_reloading()
                self.script.start_hot_reloading()
                self.ui.start_hot_reloading()
            else:
                self.editor.stop_hot_reloading()
                self.script.stop_hot_reloading()
                self.ui.stop_hot_reloading()
            logger.info(f"Editor mode: {self.editor_mode}")
        except Exception as e:
            logger.error(f"Failed to toggle editor mode: {e}")

    def load_scene(self, path):
        """Load a scene with transition."""
        try:
            self.scene.transition_out()
            self.editor.load_scene(path)
            self.scene.entities = self.editor.entities
            self.scene.transition_in()
            self.quadtree = QuadTree([0, 0, 1000, 1000])
            for entity in self.scene.entities:
                transform = entity.get_component("Transform")
                if transform:
                    self.quadtree.insert(entity, transform.position)
            logger.info(f"Loaded scene: {path}")
        except Exception as e:
            logger.error(f"Failed to load scene {path}: {e}")

    def save_scene(self, path):
        """Save the current scene."""
        try:
            self.editor.save_scene(path)
            logger.info(f"Saved scene: {path}")
        except Exception as e:
            logger.error(f"Failed to save scene {path}: {e}")

    def play_sound(self, sound_name, position=None):
        """Play a 3D sound with Doppler effect."""
        try:
            self.audio.play_sound(sound_name, position, doppler=True)
            logger.info(f"Playing sound: {sound_name}")
        except Exception as e:
            logger.error(f"Failed to play sound {sound_name}: {e}")

    def set_physics_gravity(self, gravity):
        """Set gravity for 2D and 3D physics."""
        try:
            self.physics2d.set_gravity(gravity)
            self.physics3d.set_gravity(gravity)
            logger.info(f"Set physics gravity: {gravity}")
        except Exception as e:
            logger.error(f"Failed to set gravity: {e}")

    def network_connect(self, host, port, protocol='udp'):
        """Connect to a network server with RPC support."""
        try:
            self.network.connect(host, port, protocol)
            self.network.register_rpc("update_position", lambda entity, pos: self.set_entity_position(entity, pos))
            logger.info(f"Network connected: {host}:{port} ({protocol})")
        except Exception as e:
            logger.error(f"Network connection failed: {e}")

    def set_entity_position(self, entity_name, position):
        """Set entity position via RPC."""
        try:
            entity = self.get_entity(entity_name)
            if entity:
                transform = entity.get_component("Transform")
                if transform:
                    transform.position = np.array(position, dtype='f4')
                    self.quadtree.insert(entity, transform.position)
                    logger.info(f"Set entity {entity_name} position: {position}")
        except Exception as e:
            logger.error(f"Failed to set entity position: {e}")

    def get_performance_metrics(self):
        """Get detailed performance metrics."""
        try:
            metrics = self.profiler.get_metrics()
            metrics['memory_usage'] = self.memory.get_usage()
            metrics['bandwidth'] = self.network.get_bandwidth()
            metrics['entity_count'] = len(self.scene.entities)
            metrics['visible_entities'] = len(self.visibility.get_visible_entities(self.main_camera))
            logger.debug(f"Performance metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}

    def create_default_scene(self):
        """Create a default scene with advanced features."""
        try:
            player = self.add_entity("player", {
                "Transform": {"position": [0, 0, 2], "rotation": [0, 0, 0], "scale": 1.0},
                "MeshRenderer": {"model_name": "cube.obj"},
                "Physics3D": {"collision_layer": 1},
                "Script": {"script_path": "scripts/player.py"}
            })
            enemy = self.add_entity("enemy", {
                "Transform": {"position": [5, 0, 0], "rotation": [0, 0, 0], "scale": 1.0},
                "MeshRenderer": {"model_name": "enemy.obj"},
                "TacticalAI": {"role": "aggressive", "pathfinding": "a_star", "fsm": "patrol"},
                "Physics3D": {"collision_layer": 2}
            }, parent=player)
            light = self.add_entity("light", {
                "Transform": {"position": [0, 10, 0], "rotation": [0, 0, 0], "scale": 1.0},
                "Light": {"type": "directional", "color": [1, 1, 1], "intensity": 1.0}
            })
            self.ui.add_widget("progress_bar", "health_bar", [10, 10], [200, 20], value=100, max_value=100)
            self.ui.add_widget("button", "menu_button", [10, 50], [100, 40], text="Menu", callback=lambda: logger.info("Menu clicked"))
            self.renderer3d.add_particle_system(position=[0, 0, 0], count=200, texture_name="particle.png", compute=True)
            self.renderer3d.enable_post_effect('bloom')
            self.renderer3d.enable_post_effect('ssao')
            logger.info("Created default scene")
        except Exception as e:
            logger.error(f"Failed to create default scene: {e}")

    def remap_input(self, action, new_key):
        """Remap an input action to a new key."""
        try:
            self.input_mappings[action] = new_key
            self.input.update_mappings(self.input_mappings)
            logger.info(f"Remapped {action} to {new_key}")
        except Exception as e:
            logger.error(f"Failed to remap input {action}: {e}")

    def undo_editor_action(self):
        """Undo the last editor action."""
        try:
            self.editor.undo()
            logger.info("Performed editor undo")
        except Exception as e:
            logger.error(f"Editor undo failed: {e}")

    def redo_editor_action(self):
        """Redo the last undone editor action."""
        try:
            self.editor.redo()
            logger.info("Performed editor redo")
        except Exception as e:
            logger.error(f"Editor redo failed: {e}")

    def add_ui_animation(self, widget_name, animation_type, duration):
        """Add a UI animation to a widget."""
        try:
            self.ui.add_animation(widget_name, animation_type, duration)
            logger.info(f"Added UI animation to {widget_name}: {animation_type}")
        except Exception as e:
            logger.error(f"Failed to add UI animation to {widget_name}: {e}")