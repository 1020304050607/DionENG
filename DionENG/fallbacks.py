import numpy as np
from .constants import WINDOW_SIZE, SHADOW_MAP_SIZE, PARTICLE_COUNT_MAX

# Fallback Textures (1x1 pixel RGBA arrays for simplicity)
FALLBACK_TEXTURE_2D = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)  # White pixel for sprites
FALLBACK_TEXTURE_3D_DIFFUSE = np.array([[[128, 128, 128, 255]]], dtype=np.uint8)  # Gray for 3D models
FALLBACK_TEXTURE_NORMAL = np.array([[[128, 128, 255, 255]]], dtype=np.uint8)  # Default normal map
FALLBACK_PARTICLE_TEXTURE = np.array([[[255, 255, 255, 128]]], dtype=np.uint8)  # Semi-transparent white
FALLBACK_EDITOR_GRID = np.array([[[200, 200, 200, 255]]], dtype=np.uint8)  # Light gray for editor grid

# Fallback Models (simple cube for 3D, quad for 2D)
FALLBACK_CUBE_VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
], dtype=np.float32)
FALLBACK_CUBE_INDICES = np.array([
    0, 1, 2, 2, 3, 0,  # Back
    4, 5, 6, 6, 7, 4,  # Front
    0, 4, 7, 7, 3, 0,  # Left
    1, 5, 6, 6, 2, 1,  # Right
    3, 2, 6, 6, 7, 3,  # Top
    0, 1, 5, 5, 4, 0   # Bottom
], dtype=np.uint32)
FALLBACK_QUAD_VERTICES = np.array([
    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]
], dtype=np.float32)
FALLBACK_QUAD_INDICES = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

# Fallback Shaders (OpenGL)
FALLBACK_VERTEX_SHADER_2D = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""
FALLBACK_FRAGMENT_SHADER_2D = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D texture0;
void main() {
    FragColor = texture(texture0, TexCoord);
}
"""
FALLBACK_VERTEX_SHADER_3D = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
out vec3 Normal;
out vec2 TexCoord;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
}
"""
FALLBACK_FRAGMENT_SHADER_3D = """
#version 330 core
in vec3 Normal;
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D texture_diffuse;
uniform vec3 lightDir;
void main() {
    float diff = max(dot(normalize(Normal), normalize(lightDir)), 0.0);
    FragColor = texture(texture_diffuse, TexCoord) * diff;
}
"""
FALLBACK_SHADOW_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 lightSpaceMatrix;
uniform mat4 model;
void main() {
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
"""
FALLBACK_SHADOW_FRAGMENT_SHADER = """
#version 330 core
void main() {
    gl_FragDepth = gl_FragCoord.z;
}
"""

# Fallback OpenCL Kernels (Ray Tracing and Particles)
FALLBACK_RAY_TRACING_KERNEL = """
__kernel void ray_trace(__global float* positions, __global float* normals, __global float* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    float3 ray_dir = normalize((float3)(x / (float)width - 0.5, y / (float)height - 0.5, 1.0));
    float3 color = (float3)(0.2, 0.2, 0.2);
    output[y * width + x] = color.x;
}
"""
FALLBACK_PARTICLE_KERNEL = """
__kernel void update_particles(__global float* positions, __global float* velocities, float dt) {
    int i = get_global_id(0);
    positions[i*3] += velocities[i*3] * dt;
    positions[i*3+1] += velocities[i*3+1] * dt;
    positions[i*3+2] += velocities[i*3+2] * dt;
}
"""

# Fallback DirectX Shaders (Simulated as Strings)
FALLBACK_DX_VERTEX_SHADER = """
float4 main(float3 pos : POSITION) : SV_POSITION {
    return mul(projection, mul(view, mul(model, float4(pos, 1.0))));
}
"""
FALLBACK_DX_PIXEL_SHADER = """
float4 main() : SV_TARGET {
    return float4(0.5, 0.5, 0.5, 1.0);
}
"""

# Fallback Sounds (Placeholder Data)
FALLBACK_SOUND_EFFECT = np.array([0] * 44100, dtype=np.float32)  # Silent 1-second mono audio
FALLBACK_MUSIC = np.array([0] * 44100 * 10, dtype=np.float32)  # Silent 10-second mono audio

# Fallback UI Layouts
FALLBACK_UI_LABEL = {
    'type': 'label',
    'name': 'default_label',
    'position': [10, 10],
    'size': [200, 30],
    'text': 'Default Label',
    'font_name': 'arial',
    'font_size': 24
}
FALLBACK_UI_BUTTON = {
    'type': 'button',
    'name': 'default_button',
    'position': [10, 50],
    'size': [100, 40],
    'text': 'Button',
    'callback': lambda: print("Default button clicked")
}
FALLBACK_UI_PROGRESS_BAR = {
    'type': 'progress_bar',
    'name': 'default_progress',
    'position': [10, 90],
    'size': [200, 20],
    'value': 100,
    'max_value': 100
}

# Fallback Scripts
FALLBACK_PLAYER_SCRIPT = """
def update(entity, dt, physics, assets):
    transform = entity.get_component('Transform')
    if transform:
        transform.position[0] += 0.1 * dt  # Move right
"""
FALLBACK_ENEMY_SCRIPT = """
def update(entity, dt, physics, assets):
    transform = entity.get_component('Transform')
    if transform:
        transform.position[0] -= 0.1 * dt  # Move left
"""

# Fallback Scene (for Editor)
FALLBACK_SCENE = {
    'entities': [
        {
            'name': 'default_entity',
            'components': {
                'Transform': {'position': [0, 0, 0], 'rotation': [0, 0, 0], 'scale': 1.0},
                'MeshRenderer': {'model_name': 'cube.obj'}
            }
        }
    ]
}

# Fallback Network Packet
FALLBACK_NETWORK_PACKET = {
    'rpc_id': 0,
    'data': {'position': [0, 0, 0], 'entity': 'default_entity'}
}