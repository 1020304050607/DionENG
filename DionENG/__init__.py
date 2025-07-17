import pygame
import numpy as np
import moderngl
import logging
import os
import time
import zipfile
import json
import psutil
import pybullet as p
import pybullet_data
import watchdog.observers
import watchdog.events
import dearpygui.dearpygui as dpg
try:
    import imgui
    from imgui.integrations.pygame import PygameRenderer
except ImportError:
    imgui = None
import importlib
import ctypes
import sys
import OpenGL.GL as gl

TITLE = "DionENG"
WINDOW_SIZE = (1280, 720)
ASSET_PATH = "assets"
MOD_PATH = "mods"
TEST_PATH = "tests"
FPS = 60
FIXED_DT = 1 / 120

class Entity:
    def __init__(self, name):
        self.name = name
        self.components = {}
        self.children = []

    def add_component(self, component):
        self.components[component.__class__.__name__] = component
        component.entity = self

    def get_component(self, component_name):
        return self.components.get(component_name)

class Component:
    def __init__(self):
        self.entity = None

class Transform(Component):
    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0), scale=1.0):
        super().__init__()
        self.position = np.array(position, dtype='f4')
        self.rotation = np.array(rotation, dtype='f4')
        self.scale = float(scale)

    def get_world_matrix(self):
        matrix = np.identity(4, dtype='f4')
        matrix[3, :3] = self.position
        return matrix

class Camera(Component):
    def __init__(self, fov=60, near=0.1, far=1000.0):
        super().__init__()
        self.fov = fov
        self.near = near
        self.far = far

class Physics2D(Component):
    def __init__(self):
        super().__init__()
        self.body = None

class Physics3D(Component):
    def __init__(self):
        super().__init__()
        self.body = None

class MeshRenderer(Component):
    def __init__(self, model_name, material_name=None, shader_name="pbr"):
        super().__init__()
        self.model_name = model_name
        self.material_name = material_name
        self.shader_name = shader_name

class Material(Component):
    def __init__(self, albedo_name, metallic=0.0, roughness=0.5):
        super().__init__()
        self.albedo_name = albedo_name
        self.metallic = metallic
        self.roughness = roughness

class Light(Component):
    def __init__(self, type="point", color=(1, 1, 1), intensity=1.0, shadow_enabled=False):
        super().__init__()
        self.type = type
        self.color = np.array(color, dtype='f4')
        self.intensity = intensity
        self.shadow_enabled = shadow_enabled
        self.depth_texture = None

class Decal(Component):
    def __init__(self, texture_name, position):
        super().__init__()
        self.texture_name = texture_name
        self.position = np.array(position, dtype='f4')

class NavMeshBake(Component):
    def __init__(self, data):
        super().__init__()
        self.data = data

class Animation(Component):
    def __init__(self, name, keyframes):
        super().__init__()
        self.name = name
        self.keyframes = keyframes

class TacticalAI(Component):
    def __init__(self, role="default"):
        super().__init__()
        self.role = role

class BehaviorTree(Component):
    def __init__(self, tree):
        super().__init__()
        self.tree = tree

class VRComponent(Component):
    def __init__(self):
        super().__init__()

class SelfTrainingAI(Component):
    def __init__(self):
        super().__init__()

class Scene:
    def __init__(self, name):
        self.name = name
        self.entities = []
        self.space = None

    def add_entity(self, entity):
        self.entities.append(entity)

    def save(self, filename, snapshot=False):
        data = []
        for entity in self.entities:
            entity_data = {"name": entity.name, "components": {}}
            for comp_name, comp in entity.components.items():
                if comp_name == "Transform":
                    entity_data["components"][comp_name] = {
                        "position": list(comp.position),
                        "rotation": list(comp.rotation),
                        "scale": comp.scale
                    }
                elif comp_name == "MeshRenderer":
                    entity_data["components"][comp_name] = {
                        "model_name": comp.model_name,
                        "material_name": comp.material_name,
                        "shader_name": comp.shader_name
                    }
                elif comp_name == "Light":
                    entity_data["components"][comp_name] = {
                        "type": comp.type,
                        "color": list(comp.color),
                        "intensity": comp.intensity,
                        "shadow_enabled": comp.shadow_enabled
                    }
                elif comp_name == "Decal":
                    entity_data["components"][comp_name] = {
                        "texture_name": comp.texture_name,
                        "position": list(comp.position)
                    }
                elif comp_name == "NavMeshBake":
                    entity_data["components"][comp_name] = {"data": comp.data}
                elif comp_name == "Animation":
                    entity_data["components"][comp_name] = {
                        "name": comp.name,
                        "keyframes": comp.keyframes
                    }
                elif comp_name == "TacticalAI":
                    entity_data["components"][comp_name] = {"role": comp.role}
                elif comp_name == "BehaviorTree":
                    entity_data["components"][comp_name] = {"tree": comp.tree}
            data.append(entity_data)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            self.entities.clear()
            for entity_data in data:
                entity = Entity(entity_data["name"])
                for comp_name, comp_data in entity_data["components"].items():
                    if comp_name == "Transform":
                        entity.add_component(Transform(
                            position=tuple(comp_data["position"]),
                            rotation=tuple(comp_data["rotation"]),
                            scale=comp_data["scale"]
                        ))
                    elif comp_name == "MeshRenderer":
                        entity.add_component(MeshRenderer(
                            model_name=comp_data["model_name"],
                            material_name=comp_data.get("material_name"),
                            shader_name=comp_data.get("shader_name", "pbr")
                        ))
                    elif comp_name == "Light":
                        entity.add_component(Light(
                            type=comp_data["type"],
                            color=tuple(comp_data["color"]),
                            intensity=comp_data["intensity"],
                            shadow_enabled=comp_data["shadow_enabled"]
                        ))
                    elif comp_name == "Decal":
                        entity.add_component(Decal(
                            texture_name=comp_data["texture_name"],
                            position=tuple(comp_data["position"])
                        ))
                    elif comp_name == "NavMeshBake":
                        entity.add_component(NavMeshBake(comp_data["data"]))
                    elif comp_name == "Animation":
                        entity.add_component(Animation(
                            name=comp_data["name"],
                            keyframes=comp_data["keyframes"]
                        ))
                    elif comp_name == "TacticalAI":
                        entity.add_component(TacticalAI(comp_data["role"]))
                    elif comp_name == "BehaviorTree":
                        entity.add_component(BehaviorTree(comp_data["tree"]))
                self.add_entity(entity)