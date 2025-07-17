import gc
import tracemalloc
import numpy as np
import msgpack
from array import array
from threading import Lock
from collections import defaultdict
from .. import Component, Entity, Transform, MeshRenderer, TacticalAI, Physics3D

class MemoryPool:
    """Custom memory pool for entities and components."""
    def __init__(self, obj_type, max_size, init_func=None):
        self.obj_type = obj_type
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
        self.init_func = init_func
        self.lock = Lock()

    def allocate(self):
        """Allocate an object from the pool."""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
                self.in_use.add(id(obj))
                return obj
            if len(self.in_use) < self.max_size:
                obj = self.init_func() if self.init_func else self.obj_type()
                self.in_use.add(id(obj))
                return obj
        raise MemoryError(f"No available {self.obj_type.__name__} in pool")

    def release(self, obj):
        """Return an object to the pool."""
        with self.lock:
            obj_id = id(obj)
            if obj_id in self.in_use:
                self.in_use.remove(obj_id)
                self.pool.append(obj)

    def get_stats(self):
        """Get pool usage statistics."""
        return {
            "type": self.obj_type.__name__,
            "in_use": len(self.in_use),
            "available": len(self.pool),
            "total": len(self.in_use) + len(self.pool)
        }

class MemorySystem(Component):
    def __init__(self, max_entities=1000, max_components=10000):
        super().__init__()
        self.entity_pool = MemoryPool(Entity, max_entities, lambda: Entity(""))
        self.component_pools = {
            "Transform": MemoryPool(Transform, max_components),
            "MeshRenderer": MemoryPool(MeshRenderer, max_components),
            "TacticalAI": MemoryPool(TacticalAI, max_components),
            "Physics3D": MemoryPool(Physics3D, max_components)
        }
        self.memory_usage = defaultdict(int)  # Track memory by category
        self.tracemalloc_enabled = False
        self.lock = Lock()
        tracemalloc.start()
        self.tracemalloc_enabled = True
        gc.set_threshold(1000, 10, 10)  # Optimize GC for real-time
        print(f"Memory system initialized: max_entities={max_entities}, max_components={max_components}")

    def allocate_entity(self, name):
        """Allocate an entity from the pool."""
        with self.lock:
            entity = self.entity_pool.allocate()
            entity.name = name
            self.memory_usage["entities"] += 1
            return entity

    def allocate_component(self, component_type, **kwargs):
        """Allocate a component from the pool."""
        with self.lock:
            if component_type not in self.component_pools:
                raise ValueError(f"Unsupported component type: {component_type}")
            component = self.component_pools[component_type].allocate()
            for key, value in kwargs.items():
                setattr(component, key, value)
            self.memory_usage[component_type] += 1
            return component

    def release_entity(self, entity):
        """Release an entity and its components."""
        with self.lock:
            for component in entity.components.values():
                component_type = component.__class__.__name__
                if component_type in self.component_pools:
                    self.component_pools[component_type].release(component)
                    self.memory_usage[component_type] -= 1
            self.entity_pool.release(entity)
            self.memory_usage["entities"] -= 1
            print(f"Released entity: {entity.name}")

    def track_asset_memory(self, asset_manager):
        """Track memory usage of assets."""
        with self.lock:
            for asset_type, assets in asset_manager.assets.items():
                for name, asset in assets.items():
                    if asset_type == "models" and isinstance(asset, moderngl.VertexArray):
                        self.memory_usage[f"asset_{asset_type}"] += asset.vbo.size + asset.ibo.size
                    elif asset_type == "sounds" and isinstance(asset, pygame.mixer.Sound):
                        self.memory_usage[f"asset_{asset_type}"] += asset.get_length() * 44100 * 2  # Approx bytes
                    elif asset_type == "navmeshes":
                        self.memory_usage[f"asset_{asset_type}"] += len(msgpack.packb(asset))
                    elif asset_type == "shaders" and isinstance(asset, moderngl.Program):
                        self.memory_usage[f"asset_{asset_type}"] += 1024  # Approx shader size

    def optimize_gc(self):
        """Run garbage collection with optimized settings."""
        gc.collect()
        print("Garbage collection performed")

    def profile_memory(self):
        """Profile memory usage with tracemalloc."""
        if not self.tracemalloc_enabled:
            return {}
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("lineno")
        memory_stats = {}
        for stat in stats[:10]:  # Top 10 memory consumers
            memory_stats[str(stat.traceback)] = stat.size / 1024  # KB
        return memory_stats

    def serialize_memory_stats(self):
        """Serialize memory stats for network transmission."""
        with self.lock:
            stats = {
                "entities": self.memory_usage["entities"],
                "components": {k: v for k, v in self.memory_usage.items() if k in self.component_pools},
                "assets": {k: v / 1024 for k, v in self.memory_usage.items() if k.startswith("asset_")},
                "pools": {k: v.get_stats() for k, v in self.component_pools.items()},
                "tracemalloc": self.profile_memory()
            }
            return msgpack.packb(stats)

    def update(self, entities, asset_manager, dt):
        """Update memory system, track usage, and optimize."""
        with self.lock:
            self.memory_usage.clear()
            self.memory_usage["entities"] = len(entities)
            for entity in entities:
                for component_type in entity.components:
                    if component_type in self.component_pools:
                        self.memory_usage[component_type] += 1
            self.track_asset_memory(asset_manager)
            self.optimize_gc()

            total_memory = sum(self.memory_usage.values()) / 1024  # KB
            print(f"Memory update: entities={self.memory_usage['entities']}, "
                  f"total_memory={total_memory:.2f}KB, "
                  f"pools={[p.get_stats()['in_use'] for p in self.component_pools.values()]}")

    def render_debug(self, renderer):
        """Render debug visuals for memory usage."""
        total_memory = sum(self.memory_usage.values()) / 1024  # KB
        memory_text = f"Memory: {total_memory:.2f}KB"
        renderer.add_ui_element(
            "text", position=(10, 70, 0), content=memory_text, size=24, color=(255, 255, 255)
        )
        entity_text = f"Entities: {self.memory_usage['entities']}"
        renderer.add_ui_element(
            "text", position=(10, 90, 0), content=entity_text, size=24, color=(255, 255, 255)
        )
        print("Rendered memory debug info")