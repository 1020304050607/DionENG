import asyncio
import websockets
import msgpack
import numpy as np
from collections import defaultdict
from .. import Component, Transform, Physics3D, TacticalAI

class NetworkSystem(Component):
    def __init__(self, host="localhost", port=8765, tick_rate=60):
        super().__init__()
        self.host = host
        self.port = port
        self.tick_rate = tick_rate
        self.server = None
        self.clients = {}  # {client_id: websocket}
        self.entities = {}  # {entity_id: entity}
        self.last_states = {}  # {entity_id: last_state}
        self.client_predictions = {}  # {client_id: {entity_id: state}}
        self.ping_times = {}  # {client_id: ping_ms}
        self.is_server = False
        self.loop = asyncio.get_event_loop()
        self.last_update = 0
        print(f"Network system initialized: host={host}, port={port}, tick_rate={tick_rate}")

    async def start_server(self):
        """Start the WebSocket server."""
        self.is_server = True
        self.server = await websockets.serve(self.handle_client, self.host, self.port)
        print(f"Server started on ws://{self.host}:{self.port}")
        await self.server.wait_closed()

    async def connect_client(self):
        """Connect to the server as a client."""
        self.websocket = await websockets.connect(f"ws://{self.host}:{self.port}")
        print(f"Client connected to ws://{self.host}:{self.port}")
        asyncio.create_task(self.receive_messages())

    async def handle_client(self, websocket, path):
        """Handle a new client connection."""
        client_id = len(self.clients) + 1
        self.clients[client_id] = websocket
        self.client_predictions[client_id] = {}
        self.ping_times[client_id] = 0
        print(f"Client {client_id} connected")
        try:
            await self.receive_client_messages(client_id, websocket)
        except websockets.ConnectionClosed:
            del self.clients[client_id]
            del self.client_predictions[client_id]
            del self.ping_times[client_id]
            print(f"Client {client_id} disconnected")

    async def receive_client_messages(self, client_id, websocket):
        """Receive messages from a client."""
        async for message in websocket:
            data = msgpack.unpackb(message, raw=False)
            if data["type"] == "ping":
                self.ping_times[client_id] = (self.loop.time() - data["timestamp"]) * 1000
                await websocket.send(msgpack.packb({"type": "pong"}))
            elif data["type"] == "input":
                self.apply_client_input(client_id, data["entity_id"], data["input"])
            elif data["type"] == "prediction":
                self.client_predictions[client_id][data["entity_id"]] = data["state"]

    async def receive_messages(self):
        """Receive messages from the server (client-side)."""
        async for message in self.websocket:
            data = msgpack.unpackb(message, raw=False)
            if data["type"] == "state":
                self.apply_server_state(data["entities"])
            elif data["type"] == "pong":
                self.ping_times["self"] = (self.loop.time() - data["timestamp"]) * 1000

    def apply_client_input(self, client_id, entity_id, input):
        """Apply client input to an entity (server-side)."""
        entity = self.entities.get(entity_id)
        if entity and self.physics_system:
            transform = entity.get_component("Transform")
            physics = entity.get_component("Physics3D")
            if input["action"] == "move":
                direction = np.array(input["direction"], dtype='f4')
                self.physics_system.set_velocity(entity, list(direction * 5.0) + [0])
            elif input["action"] == "jump":
                self.physics_system.apply_force(entity, [0, 0, 300])
            elif input["action"] == "attack":
                self.physics_system.apply_force(entity, [100, 0, 0])  # Example attack
            self.client_predictions[client_id][entity_id] = {
                "position": list(transform.position),
                "velocity": list(physics.body.getBaseVelocity()[0]) if physics.body else [0, 0, 0]
            }

    def apply_server_state(self, server_entities):
        """Apply server state to client entities with interpolation."""
        for entity_id, server_state in server_entities.items():
            entity = self.entities.get(entity_id)
            if entity:
                transform = entity.get_component("Transform")
                physics = entity.get_component("Physics3D")
                if transform and physics:
                    # Interpolate position
                    current_pos = np.array(transform.position, dtype='f4')
                    server_pos = np.array(server_state["position"], dtype='f4')
                    transform.position = list(current_pos + (server_pos - current_pos) * 0.1)
                    self.physics_system.set_velocity(entity, server_state["velocity"])

    def get_entity_state(self, entity):
        """Get serializable entity state."""
        transform = entity.get_component("Transform")
        physics = entity.get_component("Physics3D")
        tactical = entity.get_component("TacticalAI")
        state = {
            "position": list(transform.position) if transform else [0, 0, 0],
            "velocity": list(physics.body.getBaseVelocity()[0]) if physics and physics.body else [0, 0, 0],
            "state": self.ai_system.blackboard.get(entity.name, {}).get("state", "idle") if tactical else "none"
        }
        return state

    def update(self, entities, physics_system, ai_system, dt):
        """Update network system, sync entities, and measure ping."""
        self.physics_system = physics_system
        self.ai_system = ai_system
        self.entities = {e.name: e for e in entities if e.get_component("Transform")}

        if self.is_server:
            # Collect entity states
            current_states = {}
            for entity_id, entity in self.entities.items():
                state = self.get_entity_state(entity)
                if entity_id not in self.last_states or state != self.last_states[entity_id]:
                    current_states[entity_id] = state
                self.last_states[entity_id] = state

            # Broadcast state to clients
            if current_states:
                message = msgpack.packb({"type": "state", "entities": current_states})
                for client_id, websocket in self.clients.items():
                    asyncio.create_task(websocket.send(message))

            # Lag compensation: rewind physics for hit detection
            for client_id, predictions in self.client_predictions.items():
                for entity_id, pred_state in predictions.items():
                    entity = self.entities.get(entity_id)
                    if entity:
                        transform = entity.get_component("Transform")
                        physics = entity.get_component("Physics3D")
                        if transform and physics:
                            original_pos = transform.position
                            transform.position = pred_state["position"]
                            self.physics_system.update([entity], dt)  # Rewind simulation
                            transform.position = original_pos

        else:
            # Client: send input and predictions
            for entity_id, entity in self.entities.items():
                if entity.get_component("Physics3D"):
                    state = self.get_entity_state(entity)
                    message = msgpack.packb({
                        "type": "prediction",
                        "entity_id": entity_id,
                        "state": state
                    })
                    asyncio.create_task(self.websocket.send(message))

            # Send ping
            message = msgpack.packb({"type": "ping", "timestamp": self.loop.time()})
            asyncio.create_task(self.websocket.send(message))

        print(f"Network update: ping={self.ping_times.get('self', 0):.2f}ms, clients={len(self.clients)}")

    def send_input(self, entity_id, action, **kwargs):
        """Send client input to server."""
        if not self.is_server:
            message = msgpack.packb({
                "type": "input",
                "entity_id": entity_id,
                "input": {"action": action, **kwargs}
            })
            asyncio.create_task(self.websocket.send(message))

    def render_debug(self, renderer):
        """Render debug visuals for ping and network state."""
        ping_text = f"Ping: {self.ping_times.get('self', 0):.2f}ms"
        renderer.add_ui_element(
            "text", position=(10, 10, 0), content=ping_text, size=24, color=(255, 255, 255)
        )
        client_count = f"Clients: {len(self.clients)}"
        renderer.add_ui_element(
            "text", position=(10, 30, 0), content=client_count, size=24, color=(255, 255, 255)
        )
        print("Rendered network debug info")