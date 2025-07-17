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

class UIWidget:
    """Base class for 2D and 3D UI widgets."""
    def __init__(self, name, position, size, anchor="top-left", is_3d=False, visible=True):
        self.name = name
        self.position = np.array(position, dtype='f4')
        self.size = np.array(size, dtype='f4') if not is_3d else np.array(size[:2], dtype='f4')
        self.anchor = anchor
        self.is_3d = is_3d
        self.visible = visible
        self.color = (255, 255, 255, 255)
        self.texture = None
        self.text = ""
        self.font = None
        self.animations = []
        self.callbacks = defaultdict(list)
        self.scale = 1.0
        self.rotation = np.array([0, 0, 0], dtype='f4') if is_3d else np.array([0], dtype='f4')
        self.shader = None

    def get_screen_position(self, screen_size, camera=None):
        """Compute screen position for 2D or 3D widgets."""
        if not self.is_3d:
            if self.anchor == "top-left":
                return self.position
            elif self.anchor == "center":
                return (np.array(screen_size) - self.size) / 2 + self.position
            elif self.anchor == "bottom-right":
                return np.array(screen_size) - self.size - self.position
            return self.position
        else:
            transform = Transform(position=self.position)
            view = np.array(camera.view_matrix, dtype='f4')
            proj = np.array(camera.projection, dtype='f4')
            pos = np.array([self.position[0], self.position[1], self.position[2], 1], dtype='f4')
            clip = proj @ view @ pos
            ndc = clip[:3] / clip[3]
            return (ndc[:2] + 1) * 0.5 * np.array(screen_size)

    def is_point_inside(self, point, screen_size, camera=None):
        """Check if a point is inside the widget."""
        pos = self.get_screen_position(screen_size, camera)
        return (pos[0] <= point[0] <= pos[0] + self.size[0] and
                pos[1] <= point[1] <= pos[1] + self.size[1])

class UIButton(UIWidget):
    """Button widget for 2D/3D UI with click and hover events."""
    def __init__(self, name, position, size, text, callback, anchor="top-left", is_3d=False):
        super().__init__(name, position, size, anchor, is_3d)
        self.text = text
        self.callback = callback
        self.hovered = False
        self.clicked = False

    def update_state(self, event, screen_size, camera=None):
        """Update button state based on input."""
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_point_inside(event.pos, screen_size, camera):
            self.clicked = True
            if self.callback:
                self.callback()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.clicked = False
        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.is_point_inside(event.pos, screen_size, camera)

class UILabel(UIWidget):
    """Label widget for 2D/3D text display."""
    def __init__(self, name, position, size, text, font_name, font_size, anchor="top-left", is_3d=False):
        super().__init__(name, position, size, anchor, is_3d)
        self.text = text
        self.font_name = font_name
        self.font_size = font_size

class UIPanel(UIWidget):
    """Panel widget for grouping 2D/3D widgets."""
    def __init__(self, name, position, size, anchor="top-left", is_3d=False):
        super().__init__(name, position, size, anchor, is_3d)
        self.widgets = []

    def add_child(self, widget):
        """Add a child widget to the panel."""
        self.widgets.append(widget)

class UIAnimation:
    """Animation for UI widget properties."""
    def __init__(self, property_name, start_value, end_value, duration, easing="linear"):
        self.property_name = property_name
        self.start_value = np.array(start_value, dtype='f4')
        self.end_value = np.array(end_value, dtype='f4')
        self.duration = duration
        self.time = 0
        self.easing = easing

    def get_value(self):
        """Compute interpolated value based on easing."""
        t = min(self.time / self.duration, 1.0)
        if self.easing == "ease_in_out":
            t = 0.5 * (1 - np.cos(np.pi * t))
        return (1 - t) * self.start_value + t * self.end_value

class UIHandler(FileSystemEventHandler):
    """Handle UI layout file changes for hot-reloading."""
    def __init__(self, ui_system):
        self.ui_system = ui_system

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            self.ui_system.reload_ui_layout(event.src_path)

class UISystem(Component):
    def __init__(self, ui_dir="ui", screen_size=(1280, 720)):
        super().__init__()
        self.ui_dir = ui_dir
        self.screen_size = np.array(screen_size, dtype='i4')
        self.widgets = {}
        self.fonts = {}
        self.event_queue = []
        self.animations = defaultdict(list)
        self.lock = Lock()
        self.observer = None
        self.renderer2d = None
        self.renderer3d = None
        self.asset_manager = None
        self.script_system = None
        self.visibility_system = None
        self.ctx = None
        self.performance_metrics = defaultdict(float)
        if not os.path.exists(ui_dir):
            os.makedirs(ui_dir)
        print(f"UI system initialized: ui_dir={ui_dir}, screen_size={screen_size}")

    def set_renderers(self, renderer2d, renderer3d, ctx):
        """Set 2D and 3D renderers and ModernGL context."""
        self.renderer2d = renderer2d
        self.renderer3d = renderer3d
        self.ctx = ctx
        print("Set 2D/3D renderers and context for UI system")

    def set_asset_manager(self, asset_manager):
        """Set the asset manager for textures and fonts."""
        self.asset_manager = asset_manager
        print("Set asset manager for UI system")

    def set_script_system(self, script_system):
        """Set the script system for UI scripting."""
        self.script_system = script_system
        print("Set script system for UI system")

    def set_visibility_system(self, visibility_system):
        """Set the visibility system for 3D UI culling."""
        self.visibility_system = visibility_system
        print("Set visibility system for UI system")

    def load_font(self, font_name, font_size):
        """Load a font for text rendering."""
        with self.lock:
            key = (font_name, font_size)
            if key not in self.fonts:
                self.fonts[key] = pygame.font.SysFont(font_name, font_size)
            return self.fonts[key]

    def create_3d_ui_shader(self):
        """Create a shader for 3D UI rendering."""
        if not self.ctx:
            return None
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                uniform mat4 mvp;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = mvp * vec4(in_position, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec2 v_texcoord;
                uniform sampler2D texture0;
                uniform vec4 color;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(texture0, v_texcoord) * color;
                }
            '''
        )

    def add_widget(self, widget_type, name, position, size, is_3d=False, **kwargs):
        """Add a 2D or 3D UI widget."""
        with self.lock:
            if widget_type == "button":
                widget = UIButton(name, position, size, kwargs.get("text", ""), kwargs.get("callback"), kwargs.get("anchor", "top-left"), is_3d)
            elif widget_type == "label":
                widget = UILabel(name, position, size, kwargs.get("text", ""), kwargs.get("font_name", "arial"), kwargs.get("font_size", 24), kwargs.get("anchor", "top-left"), is_3d)
            elif widget_type == "panel":
                widget = UIPanel(name, position, size, kwargs.get("anchor", "top-left"), is_3d)
            else:
                raise ValueError(f"Unknown widget type: {widget_type}")
            if is_3d and self.asset_manager:
                widget.shader = self.create_3d_ui_shader()
                widget.texture = self.asset_manager.get_asset("ui_texture.png", "models")
            self.widgets[name] = widget
            print(f"Added widget: {name} ({widget_type}, {'3D' if is_3d else '2D'})")

    def add_animation(self, widget_name, property_name, start_value, end_value, duration, easing="linear"):
        """Add an animation to a widget."""
        with self.lock:
            if widget_name in self.widgets:
                anim = UIAnimation(property_name, start_value, end_value, duration, easing)
                self.widgets[widget_name].animations.append(anim)
                print(f"Added animation to {widget_name}: {property_name}")

    def load_ui_layout(self, path):
        """Load a UI layout from a JSON file."""
        with self.lock:
            try:
                with open(path, 'r') as f:
                    layout = json.load(f)
                for widget_data in layout.get("widgets", []):
                    self.add_widget(
                        widget_data["type"],
                        widget_data["name"],
                        widget_data["position"],
                        widget_data["size"],
                        is_3d=widget_data.get("is_3d", False),
                        text=widget_data.get("text", ""),
                        font_name=widget_data.get("font_name", "arial"),
                        font_size=widget_data.get("font_size", 24),
                        anchor=widget_data.get("anchor", "top-left"),
                        callback=eval(widget_data.get("callback", "None"))
                    )
                    if "animations" in widget_data:
                        for anim in widget_data["animations"]:
                            self.add_animation(
                                widget_data["name"],
                                anim["property"],
                                anim["start_value"],
                                anim["end_value"],
                                anim["duration"],
                                anim.get("easing", "linear")
                            )
                print(f"Loaded UI layout: {path}")
            except Exception as e:
                print(f"Error loading UI layout {path}: {e}")

    async def load_all_ui_layouts(self):
        """Load all UI layouts in the ui directory."""
        for root, _, files in os.walk(self.ui_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    self.load_ui_layout(path)
        print(f"Loaded {len(self.widgets)} UI widgets")

    def reload_ui_layout(self, path):
        """Reload a modified UI layout."""
        with self.lock:
            self.widgets.clear()
            self.load_ui_layout(path)
            print(f"Hot-reloaded UI layout: {path}")

    def start_hot_reloading(self):
        """Start monitoring UI directory for changes."""
        if not self.observer:
            self.observer = Observer()
            self.observer.schedule(UIHandler(self), self.ui_dir, recursive=True)
            self.observer.start()
            print("Started UI hot-reloading")

    def stop_hot_reloading(self):
        """Stop hot-reloading."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            print("Stopped UI hot-reloading")

    def handle_input(self, event, camera=None):
        """Handle UI input events for 2D and 3D widgets."""
        with self.lock:
            if event.type == pygame.MOUSEBUTTONDOWN:
                for widget in self.widgets.values():
                    if widget.visible and isinstance(widget, UIButton):
                        widget.update_state(event, self.screen_size, camera)
                        if widget.clicked:
                            self.event_queue.append(("click", widget.name))
            elif event.type == pygame.MOUSEMOTION:
                for widget in self.widgets.values():
                    if widget.visible and isinstance(widget, UIButton):
                        widget.update_state(event, self.screen_size, camera)
                        if widget.hovered:
                            self.event_queue.append(("hover", widget.name))
            elif event.type == pygame.KEYDOWN:
                self.event_queue.append(("key", event.key))
            elif event.type == pygame.JOYBUTTONDOWN:
                self.event_queue.append(("gamepad_button", event.button))
            print(f"Handled UI event: {event.type}, queue_size={len(self.event_queue)}")

    def update(self, entities, camera, dt):
        """Update UI animations and script integration."""
        with self.lock:
            for widget in self.widgets.values():
                for anim in widget.animations[:]:
                    anim.time += dt
                    if anim.time >= anim.duration:
                        widget.animations.remove(anim)
                    else:
                        setattr(widget, anim.property_name, anim.get_value())
            if self.script_system:
                for entity in entities:
                    script = entity.get_component("Script")
                    if script and "ui_update" in script.state:
                        script.state["ui_update"](self.widgets)
            self.performance_metrics["update_time"] += dt
            print(f"UI update: {len(self.widgets)} widgets, {len(self.event_queue)} events")

    def render_2d(self):
        """Render 2D UI widgets."""
        if not self.renderer2d:
            return
        with self.lock:
            for widget in self.widgets.values():
                if not widget.visible or widget.is_3d:
                    continue
                pos = widget.get_screen_position(self.screen_size)
                if isinstance(widget, UILabel):
                    font = self.load_font(widget.font_name, widget.font_size)
                    surface = font.render(widget.text, True, widget.color[:3])
                    self.renderer2d.add_ui_element(
                        "texture", position=pos.tolist(), content=surface, size=widget.size.tolist()
                    )
                elif isinstance(widget, UIButton):
                    color = (100, 100, 255, 255) if widget.hovered else widget.color
                    self.renderer2d.add_ui_element(
                        "rect", position=pos.tolist(), size=widget.size.tolist(), color=color
                    )
                    font = self.load_font("arial", 24)
                    surface = font.render(widget.text, True, (0, 0, 0))
                    self.renderer2d.add_ui_element(
                        "texture", position=(pos + 5).tolist(), content=surface, size=(widget.size[0] - 10, widget.size[1] - 10)
                    )
                elif isinstance(widget, UIPanel):
                    self.renderer2d.add_ui_element(
                        "rect", position=pos.tolist(), size=widget.size.tolist(), color=widget.color
                    )
                    for child in widget.widgets:
                        if isinstance(child, UILabel):
                            font = self.load_font(child.font_name, child.font_size)
                            surface = font.render(child.text, True, child.color[:3])
                            child_pos = child.get_screen_position(self.screen_size)
                            self.renderer2d.add_ui_element(
                                "texture", position=child_pos.tolist(), content=surface, size=child.size.tolist()
                            )
            print("Rendered 2D UI")

    def render_3d(self, camera):
        """Render 3D UI widgets."""
        if not self.renderer3d or not self.ctx:
            return
        with self.lock:
            visible_widgets = []
            if self.visibility_system:
                for widget in self.widgets.values():
                    if widget.is_3d and widget.visible:
                        entity = Entity(widget.name)
                        entity.add_component(Transform(position=widget.position))
                        if widget in self.visibility_system.get_visible_entities():
                            visible_widgets.append(widget)
            else:
                visible_widgets = [w for w in self.widgets.values() if w.is_3d and w.visible]
            for widget in visible_widgets:
                pos = widget.get_screen_position(self.screen_size, camera)
                mvp = np.array(camera.projection, dtype='f4') @ np.array(camera.view_matrix, dtype='f4')
                vertices = np.array([
                    [pos[0], pos[1], 0], [pos[0] + widget.size[0], pos[1], 0],
                    [pos[0] + widget.size[0], pos[1] + widget.size[1], 0], [pos[0], pos[1] + widget.size[1], 0]
                ], dtype='f4')
                texcoords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='f4')
                indices = np.array([0, 1, 2, 2, 3, 0], dtype='i4')
                vbo = self.ctx.buffer(np.hstack([vertices, texcoords]).tobytes())
                ibo = self.ctx.buffer(indices.tobytes())
                vao = self.ctx.vertex_array(
                    widget.shader or self.create_3d_ui_shader(),
                    [(vbo, '2f 2f', 'in_position', 'in_texcoord')],
                    ibo
                )
                self.renderer3d.add_ui_element(
                    "3d_widget", vao=vao, texture=widget.texture, mvp=mvp, color=widget.color
                )
            print(f"Rendered {len(visible_widgets)} 3D UI widgets")

    def serialize_ui_state(self):
        """Serialize UI state for network transmission."""
        with self.lock:
            state = {
                name: {
                    "position": w.position.tolist(),
                    "size": w.size.tolist(),
                    "visible": w.visible,
                    "text": w.text,
                    "is_3d": w.is_3d,
                    "hovered": w.hovered if isinstance(w, UIButton) else False
                } for name, w in self.widgets.items()
            }
            return msgpack.packb(state)

    def render_debug(self, renderer2d, renderer3d):
        """Render debug visuals for 2D and 3D UI widgets."""
        with self.lock:
            for widget in self.widgets.values():
                if not widget.visible:
                    continue
                pos = widget.get_screen_position(self.screen_size)
                if widget.is_3d:
                    renderer3d.add_particle_system(
                        position=widget.position.tolist(),
                        count=1,
                        texture_name="debug_point.png",
                        lifetime=0.1,
                        velocity=(0, 0, 0)
                    )
                else:
                    renderer2d.add_ui_element(
                        "rect", position=pos.tolist(), size=widget.size.tolist(),
                        color=(255, 255, 0, 50), filled=False
                    )
                    renderer2d.add_ui_element(
                        "text", position=(pos + 5).tolist(), content=widget.name,
                        size=16, color=(255, 255, 255)
                    )
            renderer2d.add_ui_element(
                "text", position=[10, 130, 0], content=f"UI Update: {self.performance_metrics['update_time']:.2f}s",
                size=24, color=(255, 255, 255)
            )
            print("Rendered UI debug info")