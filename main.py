import pygame
import moderngl
import numpy as np
import argparse
from pathlib import Path
import sys
from typing import List, Dict, Optional
from dioneng import (
    GameEngine, Scene, Entity, Transform, Sprite, MeshRenderer, Camera, TacticalAI, Physics3D, Script, Light,
    InputSystem, AudioSystem, PhysicsSystem2D, PhysicsSystem3D, RenderSystem2D, RenderSystem3D,
    AISystem, NetworkSystem, AssetManager, MemorySystem, VisibilitySystem, ScriptSystem,
    UISystem, EditorSystem, JobSystem, Profiler, Timer, Logger, EventBus, FileUtils,
    MathUtils, MathLib, WINDOW_SIZE, FALLBACK_SCENE, FALLBACK_TEXTURE_3D_DIFFUSE
)

class EngineRunner:
    """Main runner for the DionENG engine, handling initialization, loop, and shutdown."""
    def __init__(self, scene_path: str = "examples/demo_scene_1/assets/scene.json", 
                 debug: bool = False, fps_cap: int = 60, resolution: Optional[tuple] = None):
        self.engine = GameEngine()
        self.logger = Logger(name="EngineRunner")
        self.event_bus = EventBus()
        self.timer = Timer()
        self.job_system = JobSystem(max_workers=4)
        self.profiler = Profiler()
        self.file_utils = FileUtils(base_path=str(Path(__file__).parent / "assets"))
        self.math = MathLib()
        self.scene: Optional[Scene] = None
        self.entities: List[Entity] = []
        self.window = None
        self.ctx = None
        self.running = False
        self.debug = debug
        self.fps_cap = fps_cap
        self.resolution = resolution if resolution else WINDOW_SIZE
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Pygame, OpenGL, engine systems, and scene."""
        pygame.init()
        pygame.display.set_mode(self.resolution, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.window = pygame.display.get_surface()
        pygame.display.set_caption("DionENG")
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.logger.info(f"Initialized Pygame with resolution {self.resolution}")

        # Initialize systems
        self.engine.add_system(InputSystem(event_bus=self.event_bus))
        self.engine.add_system(AudioSystem())
        self.engine.add_system(PhysicsSystem2D())
        self.engine.add_system(PhysicsSystem3D())
        self.engine.add_system(RenderSystem2D())
        self.engine.add_system(RenderSystem3D(self.ctx))
        self.engine.add_system(AISystem())
        self.engine.add_system(NetworkSystem())
        self.engine.add_system(AssetManager(file_utils=self.file_utils))
        self.engine.add_system(MemorySystem())
        self.engine.add_system(VisibilitySystem())
        self.engine.add_system(ScriptSystem())
        self.engine.add_system(UISystem())
        self.engine.add_system(EditorSystem())
        self.profiler.set_job_system(self.job_system)
        
        # Load scene
        self._load_scene(self.scene_path)
        
        # Set up event listeners
        self.event_bus.subscribe("scene_updated", self._on_scene_updated)
        self.event_bus.subscribe("engine_error", self._on_engine_error)
        self.event_bus.subscribe("resize_window", self._on_resize_window)
        self.running = True
        self.logger.info("EngineRunner initialized successfully")

    def _load_scene(self, scene_path: str) -> None:
        """Load a scene from a JSON file or use fallback."""
        try:
            scene_data = self.file_utils.load_scene(scene_path)
            self.scene = Scene()
            self._create_entities_from_scene(scene_data)
            self.engine.set_scene(self.scene)
            self.logger.info(f"Loaded scene from {scene_path}")
        except Exception as e:
            self.logger.error(f"Failed to load scene {scene_path}: {e}")
            self._create_fallback_scene()

    def _create_entities_from_scene(self, scene_data: Dict) -> None:
        """Create entities from scene data."""
        for entity_data in scene_data.get("entities", []):
            entity = Entity(name=entity_data.get("name", "Entity"), tags=entity_data.get("tags", []))
            for comp_type, comp_data in entity_data.get("components", {}).items():
                component_map = {
                    "Transform": Transform,
                    "MeshRenderer": MeshRenderer,
                    "Camera": Camera,
                    "Physics3D": Physics3D,
                    "TacticalAI": TacticalAI,
                    "Light": Light,
                    "Sprite": Sprite,
                    "Script": Script
                }
                if comp_type in component_map:
                    entity.add_component(component_map[comp_type](**comp_data))
            self.scene.add_entity(entity)
            self.entities.append(entity)
        self.logger.info(f"Created {len(self.entities)} entities from scene")

    def _create_fallback_scene(self) -> None:
        """Create a fallback scene with basic entities."""
        self.scene = Scene()
        player = Entity(name="Player", tags=["player"])
        player.add_component(Transform(position=np.array([0, 0, 2], dtype=np.float32)))
        player.add_component(MeshRenderer(model_name="cube.obj", material=FALLBACK_TEXTURE_3D_DIFFUSE))
        player.add_component(Physics3D(mass=1.0))
        player.add_component(Camera(fov=60.0, near=0.1, far=1000.0, aspect_ratio=self.resolution[0]/self.resolution[1]))
        self.scene.add_entity(player)
        self.entities.append(player)
        
        enemy = Entity(name="Enemy", tags=["enemy"])
        enemy.add_component(Transform(position=np.array([5, 0, 0], dtype=np.float32)))
        enemy.add_component(MeshRenderer(model_name="sphere.obj"))
        enemy.add_component(TacticalAI(role="enemy", target=np.array([0, 0, 2], dtype=np.float32)))
        self.scene.add_entity(enemy)
        self.entities.append(enemy)
        
        light = Entity(name="Light", tags=["light"])
        light.add_component(Transform(position=np.array([0, 10, 0], dtype=np.float32)))
        light.add_component(Light(type="directional", color=np.array([1, 1, 1], dtype=np.float32), intensity=1.0))
        self.scene.add_entity(light)
        self.entities.append(light)
        
        self.engine.set_scene(self.scene)
        self.logger.info("Created fallback scene with player, enemy, and light")

    def _on_scene_updated(self, data: Dict) -> None:
        """Handle scene update events."""
        self.logger.debug(f"Scene updated: {data}")

    def _on_engine_error(self, data: Dict) -> None:
        """Handle engine error events."""
        self.logger.error(f"Engine error: {data}")
        self.running = False

    def _on_resize_window(self, data: Dict) -> None:
        """Handle window resize events."""
        new_size = data.get("size", WINDOW_SIZE)
        self.resolution = new_size
        pygame.display.set_mode(self.resolution, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.ctx.viewport = (0, 0, self.resolution[0], self.resolution[1])
        for entity in self.entities:
            if camera := entity.get_component("Camera"):
                camera.aspect_ratio = self.resolution[0] / self.resolution[1]
        self.logger.info(f"Window resized to {self.resolution}")

    def run(self) -> None:
        """Main game loop with optimized system updates."""
        clock = pygame.time.Clock()
        while self.running:
            dt = self.timer.tick()
            self.profiler.start_frame()
            
            # Handle input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_w:
                        self.event_bus.publish("player_move", {"position": np.array([0, 0, -0.1], dtype=np.float32)})
                    elif event.key == pygame.K_s:
                        self.event_bus.publish("player_move", {"position": np.array([0, 0, 0.1], dtype=np.float32)})
                elif event.type == pygame.VIDEORESIZE:
                    self.event_bus.publish("resize_window", {"size": (event.w, event.h)})
            
            # Update systems in parallel where possible
            system_jobs = [
                ("InputSystem", lambda: self.engine.get_system("InputSystem").update(dt)),
                ("PhysicsSystem3D", lambda: self.engine.get_system("PhysicsSystem3D").update(self.entities, dt)),
                ("RenderSystem3D", lambda: self.engine.get_system("RenderSystem3D").update(self.entities, dt)),
                ("AISystem", lambda: self.engine.get_system("AISystem").update(self.entities, dt)),
                ("AssetManager", lambda: self.engine.get_system("AssetManager").update(dt)),
                ("MemorySystem", lambda: self.engine.get_system("MemorySystem").update(dt)),
                ("VisibilitySystem", lambda: self.engine.get_system("VisibilitySystem").update(dt)),
                ("ScriptSystem", lambda: self.engine.get_system("ScriptSystem").update(dt)),
                ("UISystem", lambda: self.engine.get_system("UISystem").update(dt)),
                ("EditorSystem", lambda: self.engine.get_system("EditorSystem").update(dt))
            ]
            for system_name, system_func in system_jobs:
                self.profiler.profile_system(system_name, system_func)

            # Specialized profiling tasks
            player = next((e for e in self.entities if "player" in e.tags), None)
            if player:
                self.profiler.profile_physics(self.entities, dt)
                enemy = next((e for e in self.entities if "enemy" in e.tags), None)
                if enemy and (ai := enemy.get_component("TacticalAI")):
                    self.profiler.profile_ai(enemy, np.zeros((10, 10), dtype=np.float32), ai.target)
            
            # Ray tracing and quadtree demos
            rays = np.array([[[0, 0, 2], [0, 0, -1]]], dtype=np.float32)
            self.profiler.profile_ray_tracing(rays, self.entities)
            bounds = np.array([[-20.0, -20.0], [20.0, 20.0]], dtype=np.float32)
            self.job_system.run_job(lambda: self.profiler.profile_quadtree_query(self.entities, bounds))
            
            # Dynamic resolution scaling
            self.profiler.profile_dynamic_resolution(self.resolution, 1.0)
            
            # Log metrics in debug mode
            if self.debug:
                self.profiler.log_metrics()
                self.job_system.run_job(lambda: self.profiler.generate_report("profiler_report.json"))
            
            # Render frame
            self.ctx.clear(0.1, 0.1, 0.1, 1.0)
            self.engine.get_system("RenderSystem3D").render()
            pygame.display.flip()
            clock.tick(self.fps_cap)
        
        self._shutdown()

    def _shutdown(self) -> None:
        """Shutdown the engine gracefully."""
        self.running = False
        self.job_system.shutdown()
        self.profiler.shutdown()
        self.engine.shutdown()
        pygame.quit()
        self.logger.info("EngineRunner shutdown complete")

def main():
    """Parse command-line arguments and run the engine."""
    parser = argparse.ArgumentParser(description="DionENG Game Engine")
    parser.add_argument("--scene", type=str, default="examples/demo_scene_1/assets/scene.json",
                        help="Path to the scene JSON file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with profiling and logging")
    parser.add_argument("--fps", type=int, default=60,
                        help="Set FPS cap (default: 60)")
    parser.add_argument("--resolution", type=int, nargs=2, default=None,
                        help="Set window resolution (e.g., --resolution 1280 720)")
    args = parser.parse_args()
    
    runner = EngineRunner(
        scene_path=args.scene,
        debug=args.debug,
        fps_cap=args.fps,
        resolution=tuple(args.resolution) if args.resolution else None
    )
    try:
        runner.run()
    except Exception as e:
        runner.logger.error(f"Engine crashed: {e}")
        runner._shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()