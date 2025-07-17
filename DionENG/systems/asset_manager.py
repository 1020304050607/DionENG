import asyncio
import pygame
import moderngl
import numpy as np
import msgpack
import json
import os
import zlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .. import Component

class AssetHandler(FileSystemEventHandler):
    """Handle file changes for hot-reloading."""
    def __init__(self, asset_manager):
        self.asset_manager = asset_manager

    def on_modified(self, event):
        if not event.is_directory:
            self.asset_manager.reload_asset(event.src_path)

class AssetManager(Component):
    def __init__(self, asset_dir="assets"):
        super().__init__()
        self.asset_dir = asset_dir
        self.assets = {
            "models": {},  # 3D models (moderngl VAOs) and 2D textures (pygame Surfaces)
            "sounds": {},  # pygame Sound objects
            "shaders": {},  # moderngl Programs
            "navmeshes": {}  # JSON navmeshes
        }
        self.ref_counts = defaultdict(int)  # Reference counting for assets
        self.ctx = None  # ModernGL context
        self.load_tasks = []
        self.observer = None
        self.loop = asyncio.get_event_loop()
        print(f"Asset manager initialized: asset_dir={asset_dir}")

    def set_context(self, ctx):
        """Set ModernGL context for 3D assets."""
        self.ctx = ctx
        print("ModernGL context set for asset manager")

    async def load_model(self, path):
        """Load a 3D model (OBJ) or 2D texture (PNG/JPG)."""
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)
        try:
            if ext == ".obj":
                if not self.ctx:
                    raise ValueError("ModernGL context not set")
                vertices, indices = self.parse_obj(path)
                vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
                ibo = self.ctx.buffer(indices.astype('i4').tobytes())
                vao = self.ctx.vertex_array(
                    self.assets["shaders"].get("default", self.create_default_shader()),
                    [(vbo, '3f', 'in_position')],
                    ibo
                )
                self.assets["models"][name] = vao
            elif ext in (".png", ".jpg"):
                surface = pygame.image.load(path)
                self.assets["models"][name] = surface
            self.ref_counts[name] += 1
            print(f"Loaded model/texture: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            self.assets["models"][name] = self.create_fallback_model()

    async def load_sound(self, path):
        """Load a sound file (WAV)."""
        name = os.path.basename(path)
        try:
            sound = pygame.mixer.Sound(path)
            self.assets["sounds"][name] = sound
            self.ref_counts[name] += 1
            print(f"Loaded sound: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            self.assets["sounds"][name] = pygame.mixer.Sound(buffer=b"\0" * 1024)

    async def load_shader(self, path):
        """Load a shader file (GLSL)."""
        name = os.path.basename(path)
        try:
            with open(path, 'r') as f:
                vertex_shader = f.read()
            fragment_path = os.path.splitext(path)[0] + ".frag"
            with open(fragment_path, 'r') as f:
                fragment_shader = f.read()
            if not self.ctx:
                raise ValueError("ModernGL context not set")
            program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            self.assets["shaders"][name] = program
            self.ref_counts[name] += 1
            print(f"Loaded shader: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            self.assets["shaders"][name] = self.create_default_shader()

    async def load_navmesh(self, path):
        """Load a navmesh file (JSON) with compression."""
        name = os.path.basename(path)
        try:
            with open(path, 'rb') as f:
                compressed = f.read()
                data = json.loads(zlib.decompress(compressed).decode('utf-8'))
            self.assets["navmeshes"][name] = data
            self.ref_counts[name] += 1
            print(f"Loaded navmesh: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            self.assets["navmeshes"][name] = {"nodes": [], "edges": {}}

    def parse_obj(self, path):
        """Parse OBJ file for vertices and indices."""
        vertices = []
        indices = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.split()[1:4])))
                elif line.startswith('f '):
                    indices.extend([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
        return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4')

    def create_default_shader(self):
        """Create a fallback shader."""
        if not self.ctx:
            return None
        return self.ctx.program(
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
                    fragColor = vec4(1.0, 0.0, 1.0, 1.0); // Magenta fallback
                }
            '''
        )

    def create_fallback_model(self):
        """Create a fallback cube model."""
        if not self.ctx:
            return pygame.Surface((32, 32))
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype='f4')
        indices = np.array([
            0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4,
            0, 4, 7, 7, 3, 0, 1, 5, 6, 6, 2, 1,
            3, 2, 6, 6, 7, 3, 0, 1, 5, 5, 4, 0
        ], dtype='i4')
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        return self.ctx.vertex_array(
            self.create_default_shader(),
            [(vbo, '3f', 'in_position')],
            ibo
        )

    async def load_asset(self, path):
        """Load an asset based on its extension."""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".obj", ".png", ".jpg"):
            await self.load_model(path)
        elif ext == ".wav":
            await self.load_sound(path)
        elif ext == ".vert":
            await self.load_shader(path)
        elif ext == ".json":
            await self.load_navmesh(path)
        else:
            print(f"Unsupported asset type: {ext}")

    async def load_all_assets(self):
        """Load all assets in the asset directory asynchronously."""
        if not os.path.exists(self.asset_dir):
            os.makedirs(self.asset_dir)
        for root, _, files in os.walk(self.asset_dir):
            for file in files:
                path = os.path.join(root, file)
                self.load_tasks.append(asyncio.create_task(self.load_asset(path)))
        await asyncio.gather(*self.load_tasks)
        print(f"Loaded {sum(len(v) for v in self.assets.values())} assets")

    def get_asset(self, name, asset_type):
        """Get an asset with reference counting."""
        if name in self.assets[asset_type]:
            self.ref_counts[name] += 1
            return self.assets[asset_type][name]
        print(f"Asset not found: {name} ({asset_type})")
        return self.assets[asset_type].get(name, self.create_fallback_model() if asset_type == "models" else None)

    def release_asset(self, name, asset_type):
        """Release an asset and unload if no references remain."""
        if name in self.assets[asset_type]:
            self.ref_counts[name] -= 1
            if self.ref_counts[name] <= 0:
                asset = self.assets[asset_type].pop(name)
                if asset_type == "models" and isinstance(asset, moderngl.VertexArray):
                    asset.release()
                del self.ref_counts[name]
                print(f"Unloaded asset: {name} ({asset_type})")

    def reload_asset(self, path):
        """Reload a modified asset for hot-reloading."""
        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        if name in self.ref_counts:
            asyncio.create_task(self.load_asset(path))
            print(f"Hot-reloaded asset: {name}")

    def start_hot_reloading(self):
        """Start monitoring asset directory for changes."""
        if not self.observer:
            self.observer = Observer()
            self.observer.schedule(AssetHandler(self), self.asset_dir, recursive=True)
            self.observer.start()
            print("Started hot-reloading for assets")

    def stop_hot_reloading(self):
        """Stop hot-reloading."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            print("Stopped hot-reloading")

    def serialize_assets(self):
        """Serialize asset metadata for network transmission."""
        metadata = {
            "models": list(self.assets["models"].keys()),
            "sounds": list(self.assets["sounds"].keys()),
            "shaders": list(self.assets["shaders"].keys()),
            "navmeshes": list(self.assets["navmeshes"].keys())
        }
        return msgpack.packb(metadata)

    def update(self, dt):
        """Update asset manager (e.g., process async tasks)."""
        completed = []
        for task in self.load_tasks:
            if task.done():
                completed.append(task)
        for task in completed:
            self.load_tasks.remove(task)
        print(f"Asset manager update: {len(self.load_tasks)} tasks pending")