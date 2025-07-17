import pygame
import numpy as np
from .. import Component, Transform, MeshRenderer, Camera

class RenderSystem2D(Component):
    def __init__(self, window, assets):
        super().__init__()
        self.window = window
        self.assets = assets
        self.camera = None  # Active camera for rendering
        self.lights = []  # Store light sources
        self.particles = []  # Store particle systems
        self.ui_elements = []  # Store UI elements (text, images)
        self.background_color = (0, 0, 0)  # Default black background
        self.render_layers = {"background": [], "default": [], "foreground": [], "ui": []}  # Layered rendering
        self.post_effects = {"tint": None, "fade": 1.0}  # Post-processing effects
        self.screen_rect = pygame.Rect(0, 0, window.get_width(), window.get_height())  # For culling
        self.setup_renderer()

    def setup_renderer(self):
        """Initialize rendering settings."""
        pygame.display.set_caption("DionENG 2D Renderer")
        self.window.fill(self.background_color)
        print("2D renderer initialized with Pygame")

    def set_camera(self, camera_entity):
        """Set the active camera for rendering."""
        self.camera = camera_entity.get_component("Camera")
        if self.camera:
            print(f"Active camera set: {camera_entity.name}")
        else:
            print(f"Warning: {camera_entity.name} has no Camera component")

    def set_background_color(self, color):
        """Set the background color for the scene."""
        self.background_color = color
        print(f"Background color set to {color}")

    def add_light(self, position, radius, color=(255, 255, 255), intensity=1.0):
        """Add a 2D light source (point light)."""
        self.lights.append({
            "position": np.array(position, dtype='f4'),
            "radius": radius,
            "color": color,
            "intensity": intensity
        })
        print(f"Added light at {position} with radius {radius}")

    def add_particle_system(self, position, count, texture_name, lifetime=1.0, velocity=(0, 0)):
        """Add a particle system for effects (e.g., explosions, smoke)."""
        texture = self.assets.models.get(texture_name, pygame.Surface((10, 10)))
        particles = []
        for _ in range(count):
            particle = {
                "position": np.array(position, dtype='f4'),
                "velocity": np.array(velocity, dtype='f4') + np.random.uniform(-1, 1, 2),
                "lifetime": lifetime,
                "age": 0.0,
                "texture": texture
            }
            particles.append(particle)
        self.particles.append(particles)
        print(f"Added particle system at {position} with {count} particles")

    def add_ui_element(self, element_type, position, content, layer="ui", **kwargs):
        """Add a UI element (text or image) to the scene."""
        ui_element = {
            "type": element_type,
            "position": np.array(position, dtype='f4'),
            "content": content,
            "layer": layer,
            "kwargs": kwargs
        }
        self.ui_elements.append(ui_element)
        self.render_layers[layer].append(ui_element)
        print(f"Added UI element: {element_type} at {position}")

    def apply_post_effect(self, effect, value):
        """Apply a post-processing effect (e.g., tint, fade)."""
        if effect in self.post_effects:
            self.post_effects[effect] = value
            print(f"Applied post effect: {effect} = {value}")

    def render(self, entities, camera_entity=None):
        """Render the scene with entities, lights, particles, and UI."""
        # Clear screen
        self.window.fill(self.background_color)

        # Set camera if provided
        if camera_entity:
            self.set_camera(camera_entity)

        # Calculate camera transform
        camera_pos = np.array([0, 0], dtype='f4')
        camera_zoom = 1.0
        if self.camera:
            transform = camera_entity.get_component("Transform")
            if transform:
                camera_pos = transform.position[:2]
                camera_zoom = transform.scale

        # Update render layers
        self.render_layers["background"].clear()
        self.render_layers["default"].clear()
        self.render_layers["foreground"].clear()

        # Sort entities by layer
        for entity in entities:
            mesh = entity.get_component("MeshRenderer")
            if mesh:
                transform = entity.get_component("Transform")
                if transform and mesh.model_name in self.assets.models:
                    layer = mesh.shader_name if mesh.shader_name in self.render_layers else "default"
                    self.render_layers[layer].append((entity, transform, mesh))

        # Render layers in order
        for layer in ["background", "default", "foreground"]:
            for entity, transform, mesh in self.render_layers[layer]:
                # Apply camera transform
                screen_pos = (transform.position[:2] - camera_pos) * camera_zoom
                screen_pos = (int(screen_pos[0] + self.window.get_width() / 2),
                              int(screen_pos[1] + self.window.get_height() / 2))

                # Cull entities outside screen
                model = self.assets.models[mesh.model_name]
                entity_rect = model.get_rect(topleft=screen_pos)
                if not self.screen_rect.colliderect(entity_rect):
                    continue

                # Apply rotation and scaling
                rotated_model = pygame.transform.rotozoom(
                    model, -transform.rotation[2], transform.scale * camera_zoom
                )
                rotated_rect = rotated_model.get_rect(center=entity_rect.center)

                # Apply lighting
                light_surface = pygame.Surface(model.get_size(), pygame.SRCALPHA)
                for light in self.lights:
                    light_dist = np.linalg.norm(light["position"] - transform.position[:2])
                    if light_dist < light["radius"]:
                        intensity = light["intensity"] * (1 - light_dist / light["radius"])
                        light_color = [min(255, c * intensity) for c in light["color"]]
                        light_surface.fill(light_color + [int(255 * intensity)], special_flags=pygame.BLEND_RGBA_MULT)

                # Blit model with lighting
                self.window.blit(rotated_model, rotated_rect)
                if light_surface.get_width() > 0:
                    self.window.blit(light_surface, rotated_rect, special_flags=pygame.BLEND_RGBA_MULT)

        # Update and render particles
        for particle_system in self.particles[:]:
            for particle in particle_system[:]:
                particle["age"] += FIXED_DT
                if particle["age"] > particle["lifetime"]:
                    particle_system.remove(particle)
                    continue
                particle["position"] += particle["velocity"] * FIXED_DT
                screen_pos = (particle["position"] - camera_pos) * camera_zoom
                screen_pos = (int(screen_pos[0] + self.window.get_width() / 2),
                              int(screen_pos[1] + self.window.get_height() / 2))
                self.window.blit(particle["texture"], screen_pos)
            if not particle_system:
                self.particles.remove(particle_system)

        # Render UI elements
        for ui_element in self.render_layers["ui"]:
            if ui_element["type"] == "text":
                font = pygame.font.Font(None, ui_element["kwargs"].get("size", 36))
                text_surface = font.render(ui_element["content"], True, ui_element["kwargs"].get("color", (255, 255, 255)))
                self.window.blit(text_surface, ui_element["position"])
            elif ui_element["type"] == "image":
                image = self.assets.models.get(ui_element["content"], pygame.Surface((50, 50)))
                self.window.blit(image, ui_element["position"])

        # Apply post-processing effects
        if self.post_effects["tint"]:
            tint_surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)
            tint_surface.fill(self.post_effects["tint"])
            self.window.blit(tint_surfaceourneartint_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        if self.post_effects["fade"] < 1.0:
            fade_surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)
            fade_surface.fill((255, 255, 255, int(255 * (1 - self.post_effects["fade"]))))
            self.window.blit(fade_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        pygame.display.flip()
        print("Rendered 2D frame")