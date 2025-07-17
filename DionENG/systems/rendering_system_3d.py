import moderngl
import pygame
import numpy as np
from .. import Component, Transform, MeshRenderer, Light, Camera, Material

class RenderSystem3D(Component):
    def __init__(self, window, assets):
        super().__init__()
        self.window = window
        self.assets = assets
        self.ctx = moderngl.create_context()
        self.lights = []  # Store light components
        self.particles = []  # Store particle systems
        self.ui_elements = []  # Store UI elements
        self.render_layers = {"default": [], "transparent": [], "ui": []}  # Layered rendering
        self.post_effects = {"bloom": False, "color_grade": None}  # Post-processing
        self.depth_texture = None
        self.shadow_program = None
        self.pbr_program = None
        self.particle_program = None
        self.framebuffer = None
        self.setup_renderer()

    def setup_renderer(self):
        """Initialize ModernGL context, shaders, and framebuffers."""
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.screen_size = self.window.get_size()
        
        # PBR shader
        self.pbr_program = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord;
                uniform mat4 mvp;
                uniform mat4 model;
                out vec3 frag_pos;
                out vec3 frag_normal;
                out vec2 frag_texcoord;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    frag_pos = vec3(model * vec4(in_position, 1.0));
                    frag_normal = normalize(mat3(model) * in_normal);
                    frag_texcoord = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 frag_pos;
                in vec3 frag_normal;
                in vec2 frag_texcoord;
                out vec4 fragColor;
                uniform vec3 light_pos;
                uniform vec3 light_color;
                uniform float light_intensity;
                uniform vec3 camera_pos;
                uniform sampler2D albedo_map;
                uniform float metallic;
                uniform float roughness;
                void main() {
                    vec3 N = normalize(frag_normal);
                    vec3 L = normalize(light_pos - frag_pos);
                    vec3 V = normalize(camera_pos - frag_pos);
                    vec3 H = normalize(L + V);
                    float NdotL = max(dot(N, L), 0.0);
                    float NdotH = max(dot(N, H), 0.0);
                    vec3 albedo = texture(albedo_map, frag_texcoord).rgb;
                    vec3 diffuse = albedo * NdotL * light_color * light_intensity;
                    float specular = pow(NdotH, (1.0 - roughness) * 100.0) * metallic;
                    fragColor = vec4(diffuse + specular * light_color, 1.0);
                }
            '''
        )

        # Shadow mapping shader
        self.shadow_program = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                uniform mat4 light_mvp;
                void main() {
                    gl_Position = light_mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                void main() {}
            '''
        )

        # Particle shader
        self.particle_program = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec2 in_texcoord;
                uniform mat4 mvp;
                uniform vec3 particle_pos;
                uniform float particle_scale;
                void main() {
                    vec3 pos = in_position * particle_scale + particle_pos;
                    gl_Position = mvp * vec4(pos, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                uniform vec3 particle_color;
                void main() {
                    fragColor = vec4(particle_color, 1.0);
                }
            '''
        )

        # Setup shadow framebuffer
        self.depth_texture = self.ctx.depth_texture(self.screen_size)
        self.framebuffer = self.ctx.framebuffer(depth_attachment=self.depth_texture)
        print("3D renderer initialized with PBR, shadows, and particles")

    def set_camera(self, camera_entity):
        """Set the active camera for rendering."""
        self.camera = camera_entity.get_component("Camera")
        self.camera_transform = camera_entity.get_component("Transform")
        if self.camera and self.camera_transform:
            print(f"Active camera set: {camera_entity.name}")
        else:
            print(f"Warning: {camera_entity.name} has no Camera or Transform component")

    def add_light(self, light_entity):
        """Add a light source for rendering."""
        light = light_entity.get_component("Light")
        transform = light_entity.get_component("Transform")
        if light and transform:
            self.lights.append((light, transform))
            print(f"Added light: {light.type} at {transform.position}")

    def add_particle_system(self, position, count, texture_name, lifetime=1.0, velocity=(0, 0, 0)):
        """Add a particle system for effects (e.g., explosions)."""
        texture = self.assets.models.get(texture_name, None)
        particles = []
        for _ in range(count):
            particle = {
                "position": np.array(position, dtype='f4'),
                "velocity": np.array(velocity, dtype='f4') + np.random.uniform(-1, 1, 3),
                "lifetime": lifetime,
                "age": 0.0,
                "color": (1.0, 1.0, 1.0),
                "scale": 1.0
            }
            particles.append(particle)
        self.particles.append({"particles": particles, "texture": texture})
        print(f"Added particle system at {position} with {count} particles")

    def add_ui_element(self, element_type, position, content, **kwargs):
        """Add a UI element (text or image) using Pygame."""
        ui_element = {
            "type": element_type,
            "position": position[:2],
            "content": content,
            "kwargs": kwargs
        }
        self.render_layers["ui"].append(ui_element)
        print(f"Added UI element: {element_type} at {position}")

    def apply_post_effect(self, effect, value):
        """Apply a post-processing effect (e.g., bloom, color grading)."""
        if effect in self.post_effects:
            self.post_effects[effect] = value
            print(f"Applied post effect: {effect} = {value}")

    def get_mvp_matrix(self, transform, camera):
        """Calculate model-view-projection matrix."""
        model = transform.get_world_matrix()
        view = np.linalg.inv(self.camera_transform.get_world_matrix())
        proj = np.array([
            [1.0 / np.tan(np.radians(camera.fov / 2)), 0, 0, 0],
            [0, self.screen_size[0] / self.screen_size[1] / np.tan(np.radians(camera.fov / 2)), 0, 0],
            [0, 0, -(camera.far + camera.near) / (camera.far - camera.near), -1],
            [0, 0, -2 * camera.far * camera.near / (camera.far - camera.near), 0]
        ], dtype='f4')
        return proj @ view @ model

    def render(self, entities, camera_entity):
        """Render the 3D scene with PBR, shadows, particles, and UI."""
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        if camera_entity:
            self.set_camera(camera_entity)

        if not self.camera or not self.camera_transform:
            print("No valid camera set, skipping render")
            return

        # Update render layers
        self.render_layers["default"].clear()
        self.render_layers["transparent"].clear()
        for entity in entities:
            mesh = entity.get_component("MeshRenderer")
            if mesh and mesh.model_name in self.assets.models:
                transform = entity.get_component("Transform")
                material = entity.get_component("Material")
                layer = "transparent" if material and material.roughness > 0.8 else "default"
                self.render_layers[layer].append((entity, transform, mesh, material))

        # Render shadow maps
        self.framebuffer.use()
        self.framebuffer.clear()
        for light, light_transform in self.lights:
            if light.shadow_enabled:
                light_mvp = np.identity(4, dtype='f4')  # Simplified light projection
                self.shadow_program['light_mvp'].write(light_mvp)
                for entity, transform, mesh, _ in self.render_layers["default"]:
                    if mesh.model_name in self.assets.models:
                        model = self.assets.models[mesh.model_name]
                        self.shadow_program['light_mvp'].write(self.get_mvp_matrix(transform, light))
                        model.render(mode=moderngl.TRIANGLES)
        self.ctx.screen.use()

        # Render scene with PBR
        for layer in ["default", "transparent"]:
            self.ctx.enable(moderngl.DEPTH_TEST if layer == "default" else moderngl.BLEND)
            for entity, transform, mesh, material in self.render_layers[layer]:
                if mesh.model_name in self.assets.models:
                    model = self.assets.models[mesh.model_name]
                    self.pbr_program['mvp'].write(self.get_mvp_matrix(transform, self.camera))
                    self.pbr_program['model'].write(transform.get_world_matrix())
                    self.pbr_program['camera_pos'].write(self.camera_transform.position)
                    if material:
                        self.pbr_program['metallic'].value = material.metallic
                        self.pbr_program['roughness'].value = material.roughness
                        if material.albedo_name in self.assets.models:
                            self.assets.models[material.albedo_name].use()
                    for light, light_transform in self.lights:
                        self.pbr_program['light_pos'].write(light_transform.position)
                        self.pbr_program['light_color'].write(light.color)
                        self.pbr_program['light_intensity'].value = light.intensity
                    model.render(mode=moderngl.TRIANGLES)

        # Render particles
        for particle_system in self.particles[:]:
            for particle in particle_system["particles"][:]:
                particle["age"] += FIXED_DT
                if particle["age"] > particle["lifetime"]:
                    particle_system["particles"].remove(particle)
                    continue
                particle["position"] += particle["velocity"] * FIXED_DT
                self.particle_program['mvp'].write(self.get_mvp_matrix(
                    Transform(position=particle["position"], scale=particle["scale"]), self.camera
                ))
                self.particle_program['particle_pos'].write(particle["position"])
                self.particle_program['particle_scale'].value = particle["scale"]
                self.particle_program['particle_color'].write(particle["color"])
                if particle_system["texture"]:
                    particle_system["texture"].use()
                self.assets.models.get("particle_quad", None).render(mode=moderngl.TRIANGLES)
            if not particle_system["particles"]:
                self.particles.remove(particle_system)

        # Render UI with Pygame
        for ui_element in self.render_layers["ui"]:
            if ui_element["type"] == "text":
                font = pygame.font.Font(None, ui_element["kwargs"].get("size", 36))
                text_surface = font.render(ui_element["content"], True, ui_element["kwargs"].get("color", (255, 255, 255)))
                self.window.blit(text_surface, ui_element["position"])
            elif ui_element["type"] == "image":
                image = self.assets.models.get(ui_element["content"], pygame.Surface((50, 50)))
                self.window.blit(image, ui_element["position"])

        pygame.display.flip()
        print("Rendered 3D frame")