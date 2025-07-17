import ast
import os
import msgpack
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock
from collections import defaultdict
from .. import Component, Entity, Transform, MeshRenderer, Camera, TacticalAI, Physics3D

class Script(Component):
    """Component to attach scripts to entities."""
    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path
        self.compiled_code = None
        self.state = {}
        self.events = defaultdict(list)  # Event handlers: on_update, on_collision, etc.

class ScriptHandler(FileSystemEventHandler):
    """Handle script file changes for hot-reloading."""
    def __init__(self, script_system):
        self.script_system = script_system

    def on_modified(self, event):
        if not event.is_directory:
            self.script_system.reload_script(event.src_path)

class ScriptSystem(Component):
    def __init__(self, script_dir="scripts"):
        super().__init__()
        self.script_dir = script_dir
        self.scripts = {}  # {entity_id: Script}
        self.compiled_scripts = {}  # {path: compiled_code}
        self.safe_globals = {
            "np": np,
            "print": print,
            "get_entity": self.get_entity,
            "set_position": self.set_position,
            "apply_force": self.apply_force,
            "play_sound": self.play_sound,
            "set_visible": self.set_visible
        }
        self.observer = None
        self.lock = Lock()
        self.execution_times = defaultdict(float)
        self.error_log = []
        if not os.path.exists(script_dir):
            os.makedirs(script_dir)
        print(f"Script system initialized: script_dir={script_dir}")

    def get_entity(self, entity_id, entities):
        """Safe API to get an entity by ID."""
        return next((e for e in entities if e.name == entity_id), None)

    def set_position(self, entity, position):
        """Safe API to set entity position."""
        transform = entity.get_component("Transform") if entity else None
        if transform:
            transform.position = np.array(position, dtype='f4')

    def apply_force(self, entity, force):
        """Safe API to apply force to entity."""
        physics = entity.get_component("Physics3D") if entity else None
        if physics and self.physics_system:
            self.physics_system.apply_force(entity, force)

    def play_sound(self, sound_name):
        """Safe API to play a sound."""
        if self.asset_manager:
            sound = self.asset_manager.get_asset(sound_name, "sounds")
            if sound:
                sound.play()

    def set_visible(self, entity, visible):
        """Safe API to toggle entity visibility."""
        renderer = entity.get_component("MeshRenderer") if entity else None
        if renderer:
            renderer.visible = visible

    def compile_script(self, path):
        """Compile a script file to AST."""
        try:
            with open(path, 'r') as f:
                code = f.read()
            tree = ast.parse(code, mode='exec')
            compiled = compile(tree, path, 'exec')
            self.compiled_scripts[path] = compiled
            return compiled
        except Exception as e:
            self.error_log.append(f"Error compiling {path}: {e}")
            return None

    def load_script(self, entity, path):
        """Load and attach a script to an entity."""
        with self.lock:
            if not os.path.exists(path):
                self.error_log.append(f"Script not found: {path}")
                return
            compiled = self.compile_script(path)
            if compiled:
                script = Script(path)
                script.compiled_code = compiled
                entity.add_component(script)
                self.scripts[entity.name] = script
                print(f"Loaded script {path} for entity {entity.name}")

    async def load_all_scripts(self, entities):
        """Load all scripts in the script directory."""
        for root, _, files in os.walk(self.script_dir):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    entity = next((e for e in entities if e.name == os.path.splitext(file)[0]), None)
                    if entity:
                        self.load_script(entity, path)
        print(f"Loaded {len(self.scripts)} scripts")

    def reload_script(self, path):
        """Reload a modified script."""
        with self.lock:
            for entity_id, script in list(self.scripts.items()):
                if script.script_path == path:
                    compiled = self.compile_script(path)
                    if compiled:
                        script.compiled_code = compiled
                        script.state.clear()
                        print(f"Hot-reloaded script: {path} for entity {entity_id}")

    def start_hot_reloading(self):
        """Start monitoring script directory for changes."""
        if not self.observer:
            self.observer = Observer()
            self.observer.schedule(ScriptHandler(self), self.script_dir, recursive=True)
            self.observer.start()
            print("Started script hot-reloading")

    def stop_hot_reloading(self):
        """Stop hot-reloading."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            print("Stopped script hot-reloading")

    def execute_script(self, entity, script, event, entities, dt):
        """Execute a script's event handler."""
        if not script.compiled_code:
            return
        try:
            start_time = np.datetime64('now')
            local_vars = script.state.copy()
            local_vars.update({
                "entity": entity,
                "entities": entities,
                "dt": dt,
                "event": event
            })
            exec(script.compiled_code, self.safe_globals, local_vars)
            script.state.update(local_vars)
            self.execution_times[script.script_path] += (
                (np.datetime64('now') - start_time) / np.timedelta64(1, 'ms')
            )
        except Exception as e:
            self.error_log.append(f"Error executing {script.script_path}: {e}")

    def update(self, entities, physics_system, asset_manager, dt):
        """Update scripts for all entities."""
        self.physics_system = physics_system
        self.asset_manager = asset_manager
        with self.lock:
            for entity in entities:
                script = self.scripts.get(entity.name)
                if script:
                    self.execute_script(entity, script, "on_update", entities, dt)
            print(f"Script update: {len(self.scripts)} scripts executed, "
                  f"errors={len(self.error_log)}")

    def on_collision(self, entity1, entity2):
        """Handle collision events for scripts."""
        script = self.scripts.get(entity1.name)
        if script:
            self.execute_script(entity1, script, "on_collision", [entity1, entity2], 0)
            print(f"Collision event for {entity1.name}")

    def serialize_script_states(self):
        """Serialize script states for network transmission."""
        with self.lock:
            states = {eid: script.state for eid, script in self.scripts.items()}
            return msgpack.packb(states)

    def render_debug(self, renderer):
        """Render debug visuals for script execution."""
        y_offset = 110
        for path, time in self.execution_times.items():
            renderer.add_ui_element(
                "text", position=(10, y_offset, 0), content=f"Script {path}: {time:.2f}ms",
                size=24, color=(255, 255, 255)
            )
            y_offset += 20
        for error in self.error_log[-3:]:  # Last 3 errors
            renderer.add_ui_element(
                "text", position=(10, y_offset, 0), content=f"Error: {error}",
                size=24, color=(255, 0, 0)
            )
            y_offset += 20
        print("Rendered script debug info")