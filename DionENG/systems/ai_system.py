import pygame
import numpy as np
from heapq import heappush, heappop
from .. import Component, Transform, TacticalAI, BehaviorTree, NavMeshBake

class Node:
    """Base class for behavior tree nodes."""
    def __init__(self):
        self.children = []

    def execute(self, entity, blackboard):
        raise NotImplementedError

class Selector(Node):
    """Behavior tree selector: execute children until one succeeds."""
    def execute(self, entity, blackboard):
        for child in self.children:
            if child.execute(entity, blackboard):
                return True
        return False

class Sequence(Node):
    """Behavior tree sequence: execute children until one fails."""
    def execute(self, entity, blackboard):
        for child in self.children:
            if not child.execute(entity, blackboard):
                return False
        return True

class Action(Node):
    """Behavior tree action node."""
    def __init__(self, action_func):
        super().__init__()
        self.action_func = action_func

    def execute(self, entity, blackboard):
        return self.action_func(entity, blackboard)

class Condition(Node):
    """Behavior tree condition node."""
    def __init__(self, condition_func):
        super().__init__()
        self.condition_func = condition_func

    def execute(self, entity, blackboard):
        return self.condition_func(entity, blackboard)

class QLearning:
    """Lightweight Q-learning for self-training AI."""
    def __init__(self, state_space, action_space, learning_rate=0.1, discount=0.9, exploration=0.1):
        self.q_table = {}
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration = exploration

    def get_state_key(self, state):
        """Convert state to a hashable key."""
        return tuple(round(x, 2) for x in state)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        if np.random.random() < self.exploration:
            return np.random.choice(self.action_space)
        return self.action_space[np.argmax(self.q_table[state_key])]

    def update(self, state, action, reward, next_state):
        """Update Q-table based on experience."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.action_space))
        action_idx = self.action_space.index(action)
        q_value = self.q_table[state_key][action_idx]
        next_max = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action_idx] = q_value + self.learning_rate * (
            reward + self.discount * next_max - q_value
        )

class AISystem(Component):
    def __init__(self):
        super().__init__()
        self.navmesh = None
        self.agents = []
        self.debug_paths = {}  # Store paths for debug visualization
        self.blackboard = {}  # Shared AI state
        self.q_learners = {}  # Q-learning instances per agent
        self.action_space = ["move_to_target", "attack", "retreat", "idle"]
        self.state_space = 4  # [dist_to_target, health, enemy_health, ammo]
        print("AI system initialized with Q-learning")

    def set_navmesh(self, navmesh_entity):
        """Set the navmesh for pathfinding."""
        navmesh = navmesh_entity.get_component("NavMeshBake")
        if navmesh:
            self.navmesh = navmesh.data
            print(f"Navmesh set with {len(self.navmesh.get('nodes', []))} nodes")
        else:
            print("Warning: No NavMeshBake component found")

    def add_agent(self, entity):
        """Add an AI agent with Q-learning."""
        if entity.get_component("TacticalAI") or entity.get_component("BehaviorTree"):
            self.agents.append(entity)
            self.blackboard[entity.name] = {
                "state": "idle",
                "target": None,
                "path": [],
                "speed": 5.0,
                "vision_range": 10.0,
                "attack_range": 2.0,
                "health": 100.0,
                "ammo": 30
            }
            self.q_learners[entity.name] = QLearning(
                state_space=self.state_space,
                action_space=self.action_space,
                learning_rate=0.1,
                discount=0.9,
                exploration=0.1
            )
            print(f"Added AI agent: {entity.name} with Q-learning")

    def find_path(self, start, goal, navmesh):
        """A* pathfinding on navmesh."""
        if not navmesh or "nodes" not in navmesh or "edges" not in navmesh:
            return []

        open_list = []
        closed_set = set()
        came_from = {}
        g_score = {node: float('inf') for node in navmesh["nodes"]}
        f_score = {node: float('inf') for node in navmesh["nodes"]}
        g_score[start] = 0
        f_score[start] = np.linalg.norm(np.array(goal) - np.array(start))

        heappush(open_list, (f_score[start], start))
        while open_list:
            _, current = heappop(open_list)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            closed_set.add(current)
            for neighbor in navmesh["edges"].get(current, []):
                if neighbor in closed_set:
                    continue
                tentative_g_score = g_score[current] + np.linalg.norm(
                    np.array(navmesh["nodes"][neighbor]) - np.array(navmesh["nodes"][current])
                )
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + np.linalg.norm(
                        np.array(goal) - np.array(navmesh["nodes"][neighbor])
                    )
                    heappush(open_list, (f_score[neighbor], neighbor))
        return []

    def get_state(self, entity, blackboard, entities):
        """Get state for Q-learning."""
        transform = entity.get_component("Transform")
        target = blackboard["target"]
        if not transform or not target:
            return [0, blackboard["health"], 100, blackboard["ammo"]]
        target_transform = target.get_component("Transform")
        target_health = self.blackboard.get(target.name, {}).get("health", 100)
        dist = np.linalg.norm(transform.position[:2] - target_transform.position[:2])
        return [dist, blackboard["health"], target_health, blackboard["ammo"]]

    def detect_target(self, entity, entities, blackboard):
        """Detect targets within vision range."""
        transform = entity.get_component("Transform")
        if not transform:
            return False
        pos = transform.position[:2]
        for target in entities:
            if target == entity:
                continue
            target_transform = target.get_component("Transform")
            if target_transform:
                dist = np.linalg.norm(pos - target_transform.position[:2])
                if dist < blackboard["vision_range"]:
                    blackboard["target"] = target
                    return True
        blackboard["target"] = None
        return False

    def move_to_target(self, entity, blackboard, physics_system):
        """Move entity along path to target."""
        transform = entity.get_component("Transform")
        if not transform or not blackboard["path"]:
            return False
        next_pos = np.array(blackboard["path"][0])
        direction = next_pos - transform.position[:2]
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            blackboard["path"].pop(0)
            return len(blackboard["path"]) > 0
        direction = direction / dist if dist > 0 else direction
        physics_system.set_velocity(entity, list(direction * blackboard["speed"]) + [0])
        self.debug_paths[entity.name] = blackboard["path"]
        return True

    def attack_target(self, entity, blackboard, physics_system):
        """Attack target if in range."""
        transform = entity.get_component("Transform")
        target = blackboard["target"]
        if not transform or not target:
            return False
        target_transform = target.get_component("Transform")
        if target_transform:
            dist = np.linalg.norm(transform.position[:2] - target_transform.position[:2])
            if dist < blackboard["attack_range"] and blackboard["ammo"] > 0:
                physics_system.apply_force(target, [100, 0, 0])  # Knockback
                blackboard["ammo"] -= 1
                return True
        return False

    def retreat(self, entity, blackboard, physics_system):
        """Retreat from target if health is low."""
        transform = entity.get_component("Transform")
        target = blackboard["target"]
        if not transform or not target or blackboard["health"] > 20:
            return False
        target_transform = target.get_component("Transform")
        direction = transform.position[:2] - target_transform.position[:2]
        dist = np.linalg.norm(direction)
        direction = direction / dist if dist > 0 else direction
        physics_system.set_velocity(entity, list(direction * blackboard["speed"]) + [0])
        return True

    def build_behavior_tree(self, role):
        """Build a behavior tree based on AI role."""
        if role == "sniper":
            tree = Selector([
                Sequence([
                    Condition(lambda e, b: self.detect_target(e, self.agents, b)),
                    Action(lambda e, b: self.attack_target(e, b, self.physics_system))
                ]),
                Action(lambda e, b: b.update({"state": "idle"}))
            ])
        else:  # Aggressive AI with RL
            tree = Selector([
                Sequence([
                    Condition(lambda e, b: self.detect_target(e, self.agents, b)),
                    Action(lambda e, b: self.rl_action(e, b, self.physics_system))
                ]),
                Action(lambda e, b: b.update({"state": "idle"}))
            ])
        return tree

    def rl_action(self, entity, blackboard, physics_system):
        """Choose and execute action using Q-learning."""
        state = self.get_state(entity, blackboard, self.agents)
        action = self.q_learners[entity.name].choose_action(state)
        blackboard["state"] = action

        # Execute action
        if action == "move_to_target":
            success = self.move_to_target(entity, blackboard, physics_system)
            reward = 1.0 if success else -1.0
        elif action == "attack":
            success = self.attack_target(entity, blackboard, physics_system)
            reward = 2.0 if success else -1.0
        elif action == "retreat":
            success = self.retreat(entity, blackboard, physics_system)
            reward = 1.5 if success else -1.0
        else:  # idle
            success = True
            reward = 0.0

        # Update Q-table
        next_state = self.get_state(entity, blackboard, self.agents)
        self.q_learners[entity.name].update(state, action, reward, next_state)
        return success

    def update(self, entities, physics_system, dt):
        """Update AI agents, process behavior trees, and train RL."""
        self.physics_system = physics_system
        self.agents = [e for e in entities if e.get_component("TacticalAI") or e.get_component("BehaviorTree")]

        for entity in self.agents:
            blackboard = self.blackboard.get(entity.name, {})
            tactical = entity.get_component("TacticalAI")
            behavior_tree = entity.get_component("BehaviorTree")

            # Initialize behavior tree if needed
            if tactical and not behavior_tree:
                behavior_tree = BehaviorTree(self.build_behavior_tree(tactical.role))
                entity.add_component(behavior_tree)

            # Update health and ammo
            blackboard["health"] = max(0, blackboard["health"] - dt * 1.0)  # Example decay
            if blackboard["health"] <= 0:
                self.agents.remove(entity)
                del self.blackboard[entity.name]
                del self.q_learners[entity.name]
                continue

            # Update path if target exists
            if blackboard.get("target"):
                target_transform = blackboard["target"].get_component("Transform")
                if target_transform and self.navmesh:
                    start = tuple(entity.get_component("Transform").position[:2])
                    goal = tuple(target_transform.position[:2])
                    blackboard["path"] = self.find_path(start, goal, self.navmesh)

            # Execute behavior tree
            if behavior_tree:
                behavior_tree.tree.execute(entity, blackboard)
                self.blackboard[entity.name] = blackboard

    def render_debug(self, renderer):
        """Render debug visuals for AI paths and Q-values."""
        for entity_name, path in self.debug_paths.items():
            for i in range(len(path) - 1):
                start = np.array(list(path[i]) + [0.1])
                end = np.array(list(path[i + 1]) + [0.1])
                renderer.add_particle_system(
                    position=start,
                    count=1,
                    texture_name="debug_point.png",
                    lifetime=0.1,
                    velocity=(0, 0, 0)
                )
            # Render Q-value as UI text
            q_values = self.q_learners.get(entity_name, None)
            if q_values:
                state = self.get_state(
                    next(e for e in self.agents if e.name == entity_name),
                    self.blackboard[entity_name],
                    self.agents
                )
                q_text = f"Q: {q_values.q_table.get(q_values.get_state_key(state), [0]*len(self.action_space))}"
                renderer.add_ui_element(
                    "text", position=(10, 50), content=q_text, size=24, color=(255, 255, 255)
                )
        print("Rendered AI debug paths and Q-values")