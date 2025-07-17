import pygame
import moderngl
import numpy as np
from typing import List, Dict
from time import perf_counter
from pathlib import Path
from dioneng import (
    GameEngine, Transform, Sprite, MeshRenderer, Camera, TacticalAI, Physics3D, Script, Light,
    Entity, InputSystem, AudioSystem, PhysicsSystem2D, PhysicsSystem3D, RenderSystem2D,
    RenderSystem3D, AISystem, NetworkSystem, AssetManager, MemorySystem, VisibilitySystem,
    ScriptSystem, UISystem, EditorSystem, JobSystem, Profiler, Timer, Logger, EventBus,
    FileUtils, MathUtils, MathLib, WINDOW_SIZE, FALLBACK_SCENE, FALLBACK_TEXTURE_3D_DIFFUSE
)

class DemoScene:
    """Demo scene for DionENG showcasing ECS and systems."""
    def __init__(self):
        self.engine = GameEngine()
        self.logger = Logger(name="DemoScene")
        self.event_bus = EventBus()
        self.timer = Timer()
        self.job_system = JobSystem()
        self.profiler = Profiler()
        self.file_utils = FileUtils(base_path=str(Path(__file__).parent / "assets"))
        self.math = MathLib()
        self.entities: List[Entity] = []
        self.running = False
        self.window = None
        self.ctx = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Pygame, OpenGL, and engine systems."""
        pygame.init()
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
        self.window = pygame.display.get_surface()
        self.ctx = moderngl.create_context()
        self.logger.info(f"Initialized Pygame with window size {WINDOW_SIZE}")
        
        # Initialize systems
        self.engine.add_system(InputSystem())
        self.engine.add_system(PhysicsSystem3D())
        self.engine.add_system(RenderSystem3D(self.ctx))
        self.engine.add_system(AISystem())
        self.engine.add_system(AssetManager())
        self.engine.add_system(MemorySystem())
        self.engine.add_system(VisibilitySystem())
        self.engine.add_system(ScriptSystem())
        self.engine.add_system(UISystem())
        self.engine.add_system(EditorSystem())
        self.profiler.set_job_system(self.job_system)
        self.running = True
        
        # Load scene
        self._load_scene("scene.json")
        
        # Set up event listeners
        self.event_bus.subscribe("player_move", self._on_player_move)
        self.event_bus.subscribe("collision_detected", self._on_collision)

    def _load_scene(self, scene_path: str) -> None:
        """Load scene from JSON file or use fallback."""
        try:
            scene_data = self.file_utils.load_scene(scene_path)
            self._create_entities_from_scene(scene_data)
        except Exception as e:
            self.logger.error(f"Failed to load scene {scene_path}: {e}")
            self._create_fallback_scene()

    def _create_entities_from_scene(self, scene_data: Dict) -> None:
        """Create entities from scene data."""
        for entity_data in scene_data.get("entities", []):
            entity = Entity(name=entity_data.get("name", "Entity"), tags=entity_data.get("tags", []))
            for comp_type, comp_data in entity_data.get("components", {}).items():
                if comp_type == "Transform":
                    entity.add_component(Transform(**comp_data))
                elif comp_type == "MeshRenderer":
                    entity.add_component(MeshRenderer(**comp_data))
                elif comp_type == "Camera":
                    entity.add_component(Camera(**comp_data))
                elif comp_type == "Physics3D":
                    entity.add_component(Physics3D(**comp_data))
                elif comp_type == "TacticalAI":
                    entity.add_component(TacticalAI(**comp_data))
                elif comp_type == "Light":
                    entity.add_component(Light(**comp_data))
            self.engine.add_entity(entity)
            self.entities.append(entity)
        self.logger.info(f"Loaded {len(self.entities)} entities from scene")

    def _create_fallback_scene(self) -> None:
        """Create a fallback scene with basic entities."""
        # Player
        player = Entity(name="Player", tags=["player"])
        player.add_component(Transform(position=(0, 0, 2)))
        player.add_component(MeshRenderer(model_name="cube.obj", material=FALLBACK_TEXTURE_3D_DIFFUSE))
        player.add_component(Physics3D(mass=1.0))
        player.add_component(Camera(fov=60.0, near=0.1, far=1000.0))
        self.engine.add_entity(player)
        self.entities.append(player)
        
        # Enemy
        enemy = Entity(name="Enemy", tags=["enemy"])
        enemy.add_component(Transform(position=(5, 0, 0)))
        enemy.add_component(MeshRenderer(model_name="sphere.obj"))
        enemy.add_component(TacticalAI(role="enemy", target=np.array([0, 0, 0], dtype=np.float32)))
        self.engine.add_entity(enemy)
        self.entities.append(enemy)
        
        # Light
        light = Entity(name="Light", tags=["light"])
        light.add_component(Transform(position=(0, 10, 0)))
        light.add_component(Light(type="directional", color=(1, 1, 1), intensity=1.0))
        self.engine.add_entity(light)
        self.entities.append(light)
        
        self.logger.info("Created fallback scene with player, enemy, and light")

    def _on_player_move(self, data: Dict) -> None:
        """Handle player movement event."""
        player = next((e for e in self.entities if "player" in e.tags), None)
        if player:
            transform = player.get_component("Transform")
            if transform:
                transform.set_position(np.array(data.get("position", transform.position), dtype=np.float32))
                self.event_bus.publish("scene_updated", {"entity_id": player.id})

    def _on_collision(self, data: Dict) -> None:
        """Handle collision event."""
        self.logger.info(f"Collision detected: {data}")
        self.event_bus.publish("scene_updated", data)

    def run(self) -> None:
        """Main game loop."""
        clock = pygame.time.Clock()
        target_fps = 60
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
                        self.event_bus.publish("player_move", {"position": (0, 0, 2 - 0.1)})
                    elif event.key == pygame.K_s:
                        self.event_bus.publish("player_move", {"position": (0, 0, 2 + 0.1)})
            
            # Update systems via JobSystem
            self.profiler.profile_system("InputSystem", lambda: self.engine.get_system("InputSystem").update(dt))
            self.job_system.run_system_jobs("PhysicsSystem3D", self.entities, dt)
            self.job_system.run_system_jobs("RenderSystem3D", self.entities, dt)
            self.job_system.run_system_jobs("AISystem", self.entities, dt)
            self.profiler.profile_system("AssetManager", lambda: self.engine.get_system("AssetManager").update(dt))
            self.profiler.profile_system("MemorySystem", lambda: self.engine.get_system("MemorySystem").update(dt))
            self.profiler.profile_system("VisibilitySystem", lambda: self.engine.get_system("VisibilitySystem").update(dt))
            self.profiler.profile_system("ScriptSystem", lambda: self.engine.get_system("ScriptSystem").update(dt))
            self.profiler.profile_system("UISystem", lambda: self.engine.get_system("UISystem").update(dt))
            self.profiler.profile_system("EditorSystem", lambda: self.engine.get_system("EditorSystem").update(dt))
            
            # Profile specialized tasks
            player = next((e for e in self.entities if "player" in e.tags), None)
            if player:
                self.profiler.profile_physics(self.entities, dt)
                enemy = next((e for e in self.entities if "enemy" in e.tags), None)
                if enemy:
                    ai = enemy.get_component("TacticalAI")
                    if ai:
                        self.profiler.profile_ai(enemy, np.zeros((10, 10), dtype=np.float32), ai.target)
            
            # Ray tracing demo
            rays = np.array([[(0, 0, 2), (0, 0, -1)]], dtype=np.float32)
            self.profiler.profile_ray_tracing(rays, self.entities)
            
            # Quadtree query demo
            bounds = ((-10, -10), (10, 10))
            self.profiler.profile_quadtree_query(self.entities, bounds)
            
            # Dynamic resolution demo
            self.profiler.profile_dynamic_resolution(WINDOW_SIZE, 1.0)
            
            # Log metrics
            self.profiler.log_metrics()
            
            # Render
            self.ctx.clear(0.1, 0.1, 0.1, 1.0)
            self.engine.get_system("RenderSystem3D").render()
            pygame.display.flip()
            clock.tick(target_fps)
        
        self._shutdown()

    def _shutdown(self) -> None:
        """Shutdown the demo scene."""
        self.running = False
        self.job_system.shutdown()
        self.profiler.shutdown()
        self.engine.shutdown()
        pygame.quit()
        self.logger.info("DemoScene shutdown complete")

if __name__ == "__main__":
    demo = DemoScene()
    demo.run()