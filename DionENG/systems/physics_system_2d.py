import pybullet as p
import numpy as np
import json
from .. import Component, Transform, Physics2D

class PhysicsSystem(Component):
    def __init__(self, space, client):
        super().__init__()
        self.space = space
        self.client = client
        self.collision_callbacks = {}  # Maps collision pairs to callback functions
        self.gravity = (0, -9.81)  # Default 2D gravity (y-axis)
        self.friction = 0.5  # Default friction for CS2-like movement
        self.ground_plane_id = None
        self.map_bodies = []  # Store map collision bodies
        self.setup_physics()

    def setup_physics(self):
        """Initialize PyBullet for 2D physics with a ground plane."""
        p.setGravity(self.gravity[0], self.gravity[1], 0, physicsClientId=self.client)
        # Create a ground plane to prevent objects from falling through
        self.ground_plane_id = p.createCollisionShape(
            p.GEOM_PLANE, planeNormal=[0, 0, 1], physicsClientId=self.client
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=self.ground_plane_id,
            basePosition=[0, 0, 0], physicsClientId=self.client
        )
        print("2D physics initialized with ground plane")

    def compile_map(self, map_data):
        """Load 2D map geometry from map_data (e.g., JSON with walls, floors)."""
        try:
            if isinstance(map_data, str):
                with open(map_data, 'r') as f:
                    map_data = json.load(f)
            # Clear existing map bodies
            for body_id in self.map_bodies:
                p.removeBody(body_id, physicsClientId=self.client)
            self.map_bodies.clear()
            # Load static map objects (e.g., walls, platforms)
            for obj in map_data.get("objects", []):
                shape_type = obj.get("type", "box")
                position = obj.get("position", [0, 0])
                size = obj.get("size", [1, 1])
                if shape_type == "box":
                    shape_id = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, 0.01],
                        physicsClientId=self.client
                    )
                elif shape_type == "circle":
                    shape_id = p.createCollisionShape(
                        p.GEOM_SPHERE, radius=size[0], physicsClientId=self.client
                    )
                body_id = p.createMultiBody(
                    baseMass=0,  # Static object
                    baseCollisionShapeIndex=shape_id,
                    basePosition=[position[0], position[1], 0],
                    physicsClientId=self.client
                )
                self.map_bodies.append(body_id)
                print(f"Added map object: {shape_type} at {position}")
        except Exception as e:
            print(f"Failed to compile map: {e}")

    def add_entity_body(self, entity, shape="box", size=(1, 1), mass=1.0):
        """Add a physics body to an entity with a 2D collision shape."""
        transform = entity.get_component("Transform")
        physics = entity.get_component("Physics2D")
        if transform and physics and not physics.body:
            position = list(transform.position[:2]) + [0]  # Lock z-axis
            if shape == "box":
                shape_id = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, 0.01],
                    physicsClientId=self.client
                )
            elif shape == "circle":
                shape_id = p.createCollisionShape(
                    p.GEOM_SPHERE, radius=size[0], physicsClientId=self.client
                )
            else:
                print(f"Unsupported shape: {shape}")
                return
            physics.body = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=shape_id,
                basePosition=position,
                physicsClientId=self.client
            )
            p.changeDynamics(
                physics.body, -1, linearDamping=self.friction,
                angularDamping=0.0, physicsClientId=self.client
            )
            print(f"Added physics body to {entity.name}: {shape}, mass={mass}")

    def apply_force(self, entity, force):
        """Apply a 2D force to an entity (e.g., for jumps, explosions)."""
        physics = entity.get_component("Physics2D")
        if physics and physics.body:
            p.applyExternalForce(
                physics.body, -1, [force[0], force[1], 0],
                [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client
            )
            print(f"Applied force {force} to {entity.name}")

    def set_velocity(self, entity, velocity):
        """Set linear velocity for CS2-like movement."""
        physics = entity.get_component("Physics2D")
        if physics and physics.body:
            p.resetBaseVelocity(
                physics.body, linearVelocity=[velocity[0], velocity[1], 0],
                physicsClientId=self.client
            )
            print(f"Set velocity {velocity} for {entity.name}")

    def register_collision_callback(self, entity1_name, entity2_name, callback):
        """Register a callback for collisions between two entities."""
        self.collision_callbacks[(entity1_name, entity2_name)] = callback
        self.collision_callbacks[(entity2_name, entity1_name)] = callback  # Bidirectional
        print(f"Registered collision callback for {entity1_name} and {entity2_name}")

    def update(self, entities, dt):
        """Update physics simulation and sync entity transforms."""
        # Step physics simulation with fixed time step
        p.stepSimulation(stepTime=dt, physicsClientId=self.client)
        
        # Sync entity positions
        for entity in entities:
            if entity.get_component("Physics2D"):
                transform = entity.get_component("Transform")
                body = entity.get_component("Physics2D").body
                if body:
                    position, _ = p.getBasePositionAndOrientation(body, self.client)
                    transform.position[:2] = position[:2]  # Update x, y only

        # Check for collisions
        for entity1 in entities:
            body1 = entity1.get_component("Physics2D").body if entity1.get_component("Physics2D") else None
            if not body1:
                continue
            for entity2 in entities:
                if entity1 == entity2:
                    continue
                body2 = entity2.get_component("Physics2D").body if entity2.get_component("Physics2D") else None
                if not body2:
                    continue
                contact_points = p.getContactPoints(body1, body2, physicsClientId=self.client)
                if contact_points:
                    callback = self.collision_callbacks.get((entity1.name, entity2.name))
                    if callback:
                        callback(entity1, entity2)
                        print(f"Collision detected: {entity1.name} vs {entity2.name}")