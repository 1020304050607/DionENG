import pybullet as p
import numpy as np
import json
from .. import Component, Transform, Physics3D

class PhysicsSystem(Component):
    def __init__(self, space, client):
        super().__init__()
        self.space = space
        self.client = client
        self.collision_callbacks = {}  # Maps (entity1_name, entity2_name) to callback functions
        self.trigger_callbacks = {}  # Maps trigger_name to callback functions
        self.gravity = (0, 0, -9.81)  # Default 3D gravity (z-axis)
        self.friction = 0.5  # Friction for CS2-like movement
        self.restitution = 0.3  # Bounciness for realistic collisions
        self.max_sub_steps = 10  # For stable physics simulation
        self.map_bodies = []  # Store map collision bodies
        self.ragdoll_bodies = {}  # Store ragdoll bodies per entity
        self.constraints = []  # Store physics constraints
        self.setup_physics()

    def setup_physics(self):
        """Initialize PyBullet for 3D physics with optimized settings."""
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2], physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=FIXED_DT,
            numSubSteps=self.max_sub_steps,
            physicsClientId=self.client
        )
        # Create a ground plane for baseline collisions
        ground_shape = p.createCollisionShape(
            p.GEOM_PLANE, planeNormal=[0, 0, 1], physicsClientId=self.client
        )
        self.ground_plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.client
        )
        self.map_bodies.append(self.ground_plane_id)
        print("3D physics initialized with ground plane and fixed time step")

    def compile_map(self, map_data):
        """Load 3D map geometry from JSON or OBJ file."""
        try:
            # Clear existing map bodies
            for body_id in self.map_bodies:
                p.removeBody(body_id, physicsClientId=self.client)
            self.map_bodies.clear()
            self.map_bodies.append(self.ground_plane_id)  # Re-add ground plane

            # Load map data
            if isinstance(map_data, str):
                if map_data.endswith('.json'):
                    with open(map_data, 'r') as f:
                        map_data = json.load(f)
                elif map_data.endswith('.obj'):
                    shape_id = p.loadURDF(map_data, useFixedBase=True, physicsClientId=self.client)
                    self.map_bodies.append(shape_id)
                    print(f"Loaded map from OBJ: {map_data}")
                    return

            # Process JSON map data
            for obj in map_data.get("objects", []):
                shape_type = obj.get("type", "box")
                position = obj.get("position", [0, 0, 0])
                size = obj.get("size", [1, 1, 1])
                is_trigger = obj.get("is_trigger", False)
                if shape_type == "box":
                    shape_id = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                        physicsClientId=self.client
                    )
                elif shape_type == "sphere":
                    shape_id = p.createCollisionShape(
                        p.GEOM_SPHERE, radius=size[0], physicsClientId=self.client
                    )
                elif shape_type == "capsule":
                    shape_id = p.createCollisionShape(
                        p.GEOM_CAPSULE, radius=size[0], height=size[1],
                        physicsClientId=self.client
                    )
                else:
                    print(f"Unsupported shape type: {shape_type}")
                    continue
                body_id = p.createMultiBody(
                    baseMass=0 if is_trigger else 0,  # Triggers are static
                    baseCollisionShapeIndex=shape_id,
                    basePosition=position,
                    physicsClientId=self.client
                )
                if is_trigger:
                    p.changeDynamics(body_id, -1, collisionMargin=0.0, physicsClientId=self.client)
                    self.trigger_callbacks[body_id] = obj.get("callback", None)
                self.map_bodies.append(body_id)
                print(f"Added map object: {shape_type} at {position}, trigger={is_trigger}")
        except Exception as e:
            print(f"Failed to compile map: {e}")

    def add_entity_body(self, entity, shape="box", size=(1, 1, 1), mass=1.0, is_character=False):
        """Add a physics body to an entity (dynamic or character controller)."""
        transform = entity.get_component("Transform")
        physics = entity.get_component("Physics3D")
        if transform and physics and not physics.body:
            position = list(transform.position)
            if shape == "box":
                shape_id = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                    physicsClientId=self.client
                )
            elif shape == "sphere":
                shape_id = p.createCollisionShape(
                    p.GEOM_SPHERE, radius=size[0], physicsClientId=self.client
                )
            elif shape == "capsule":
                shape_id = p.createCollisionShape(
                    p.GEOM_CAPSULE, radius=size[0], height=size[1],
                    physicsClientId=self.client
                )
            else:
                print(f"Unsupported shape: {shape}")
                return
            if is_character:
                # Character controller: capsule with kinematic control
                physics.body = p.createMultiBody(
                    baseMass=mass,
                    baseCollisionShapeIndex=shape_id,
                    basePosition=position,
                    physicsClientId=self.client
                )
                p.changeDynamics(
                    physics.body, -1, linearDamping=self.friction, angularDamping=0.0,
                    restitution=self.restitution, physicsClientId=self.client
                )
                p.setCollisionFilterGroupMask(
                    physics.body, -1, collisionFilterGroup=1, collisionFilterMask=1,
                    physicsClientId=self.client
                )
            else:
                # Dynamic object
                physics.body = p.createMultiBody(
                    baseMass=mass,
                    baseCollisionShapeIndex=shape_id,
                    basePosition=position,
                    physicsClientId=self.client
                )
                p.changeDynamics(
                    physics.body, -1, linearDamping=self.friction, angularDamping=0.0,
                    restitution=self.restitution, physicsClientId=self.client
                )
            print(f"Added physics body to {entity.name}: {shape}, mass={mass}, character={is_character}")

    def add_ragdoll(self, entity, bones):
        """Add a ragdoll to an entity with joint constraints."""
        transform = entity.get_component("Transform")
        physics = entity.get_component("Physics3D")
        if transform and physics and not physics.body:
            position = list(transform.position)
            ragdoll_bodies = []
            for bone in bones:
                shape_id = p.createCollisionShape(
                    p.GEOM_CAPSULE, radius=bone.get("radius", 0.1), height=bone.get("height", 0.5),
                    physicsClientId=self.client
                )
                bone_position = [position[i] + bone.get("offset", [0, 0, 0])[i] for i in range(3)]
                body_id = p.createMultiBody(
                    baseMass=bone.get("mass", 1.0),
                    baseCollisionShapeIndex=shape_id,
                    basePosition=bone_position,
                    physicsClientId=self.client
                )
                ragdoll_bodies.append(body_id)
            # Create constraints between bones (e.g., hinges)
            for i in range(len(bones) - 1):
                constraint = p.createConstraint(
                    ragdoll_bodies[i], -1, ragdoll_bodies[i+1], -1,
                    p.CONSTRAINT_HINGE, pivotInA=[0, 0, 0], pivotInB=[0, 0, 0],
                    axisInA=[0, 0, 1], axisInB=[0, 0, 1], physicsClientId=self.client
                )
                self.constraints.append(constraint)
            physics.body = ragdoll_bodies[0]  # Main body for transform syncing
            self.ragdoll_bodies[entity.name] = ragdoll_bodies
            print(f"Added ragdoll to {entity.name} with {len(bones)} bones")

    def apply_force(self, entity, force, position=None):
        """Apply a 3D force to an entity (e.g., for explosions, recoil)."""
        physics = entity.get_component("Physics3D")
        if physics and physics.body:
            pos = position if position else [0, 0, 0]
            p.applyExternalForce(
                physics.body, -1, force, pos, p.WORLD_FRAME, physicsClientId=self.client
            )
            print(f"Applied force {force} to {entity.name} at {pos}")

    def set_velocity(self, entity, velocity):
        """Set linear velocity for CS2-like movement."""
        physics = entity.get_component("Physics3D")
        if physics and physics.body:
            p.resetBaseVelocity(
                physics.body, linearVelocity=velocity, physicsClientId=self.client
            )
            print(f"Set velocity {velocity} for {entity.name}")

    def add_constraint(self, entity1, entity2, constraint_type="point", pivot1=[0, 0, 0], pivot2=[0, 0, 0]):
        """Add a physics constraint between two entities (e.g., hinge, slider)."""
        physics1 = entity1.get_component("Physics3D")
        physics2 = entity2.get_component("Physics3D")
        if physics1 and physics2 and physics1.body and physics2.body:
            if constraint_type == "point":
                constraint = p.createConstraint(
                    physics1.body, -1, physics2.body, -1,
                    p.CONSTRAINT_POINT2POINT, pivotInA=pivot1, pivotInB=pivot2,
                    physicsClientId=self.client
                )
            elif constraint_type == "hinge":
                constraint = p.createConstraint(
                    physics1.body, -1, physics2.body, -1,
                    p.CONSTRAINT_HINGE, pivotInA=pivot1, pivotInB=pivot2,
                    axisInA=[0, 0, 1], axisInB=[0, 0, 1], physicsClientId=self.client
                )
            elif constraint_type == "slider":
                constraint = p.createConstraint(
                    physics1.body, -1, physics2.body, -1,
                    p.CONSTRAINT_SLIDER, pivotInA=pivot1, pivotInB=pivot2,
                    axisInA=[1, 0, 0], axisInB=[1, 0, 0], physicsClientId=self.client
                )
            else:
                print(f"Unsupported constraint type: {constraint_type}")
                return
            self.constraints.append(constraint)
            print(f"Added {constraint_type} constraint between {entity1.name} and {entity2.name}")

    def register_collision_callback(self, entity1_name, entity2_name, callback):
        """Register a callback for collisions between two entities."""
        self.collision_callbacks[(entity1_name, entity2_name)] = callback
        self.collision_callbacks[(entity2_name, entity1_name)] = callback  # Bidirectional
        print(f"Registered collision callback for {entity1_name} and {entity2_name}")

    def register_trigger_callback(self, trigger_id, callback):
        """Register a callback for trigger volume events."""
        self.trigger_callbacks[trigger_id] = callback
        print(f"Registered trigger callback for trigger ID {trigger_id}")

    def update(self, entities, dt):
        """Update physics simulation, sync transforms, and handle collisions/triggers."""
        # Step physics simulation
        p.stepSimulation(stepTime=dt, physicsClientId=self.client)

        # Sync entity transforms
        for entity in entities:
            if entity.get_component("Physics3D"):
                transform = entity.get_component("Transform")
                body = entity.get_component("Physics3D").body
                if body:
                    position, orientation = p.getBasePositionAndOrientation(body, self.client)
                    transform.position = position  # Update x, y, z
                    # Convert quaternion to Euler angles for rotation
                    transform.rotation = p.getEulerFromQuaternion(orientation)

        # Check for collisions and triggers
        for entity1 in entities:
            body1 = entity1.get_component("Physics3D").body if entity1.get_component("Physics3D") else None
            if not body1:
                continue
            # Entity-entity collisions
            for entity2 in entities:
                if entity1 == entity2:
                    continue
                body2 = entity2.get_component("Physics3D").body if entity2.get_component("Physics3D") else None
                if not body2:
                    continue
                contact_points = p.getContactPoints(body1, body2, physicsClientId=self.client)
                if contact_points:
                    callback = self.collision_callbacks.get((entity1.name, entity2.name))
                    if callback:
                        callback(entity1, entity2)
                        print(f"Collision detected: {entity1.name} vs {entity2.name}")
            # Entity-map collisions
            for map_body in self.map_bodies:
                contact_points = p.getContactPoints(body1, map_body, physicsClientId=self.client)
                if contact_points and map_body in self.trigger_callbacks:
                    callback = self.trigger_callbacks.get(map_body)
                    if callback:
                        callback(entity1)
                        print(f"Trigger activated: {entity1.name} in trigger {map_body}")