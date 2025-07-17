import pygame
from .. import Component

class InputSystem(Component):
    def __init__(self):
        super().__init__()
        self.actions = {}
        self.keys = {
            # Movement (FPS, Platformer, RPG)
            "move_forward": pygame.K_w,
            "move_backward": pygame.K_s,
            "move_left": pygame.K_a,
            "move_right": pygame.K_d,
            "jump": pygame.K_SPACE,
            "crouch": pygame.K_c,
            "sprint": pygame.K_LSHIFT,
            "dodge": pygame.K_LALT,
            "move_up": pygame.K_r,  # For 3D movement (e.g., flight sim)
            "move_down": pygame.K_f,
            # Alternative movement (arrows for accessibility)
            "move_forward_alt": pygame.K_UP,
            "move_backward_alt": pygame.K_DOWN,
            "move_left_alt": pygame.K_LEFT,
            "move_right_alt": pygame.K_RIGHT,
            # Combat (FPS, Action)
            "attack": pygame.K_LCTRL,
            "reload": pygame.K_r,
            "switch_weapon_next": pygame.K_e,
            "switch_weapon_prev": pygame.K_q,
            "ability_1": pygame.K_1,
            "ability_2": pygame.K_2,
            "ability_3": pygame.K_3,
            "ability_4": pygame.K_4,
            "melee": pygame.K_f,
            # Camera (FPS, Third-Person, RTS)
            "look_up": pygame.K_i,
            "look_down": pygame.K_k,
            "look_left": pygame.K_j,
            "look_right": pygame.K_l,
            "zoom_in": pygame.K_PLUS,
            "zoom_out": pygame.K_MINUS,
            "free_look": pygame.K_LALT,
            # UI/Editor (General, Editor)
            "open_menu": pygame.K_ESCAPE,
            "open_inventory": pygame.K_i,
            "save_game": pygame.K_F5,
            "load_game": pygame.K_F9,
            "toggle_console": pygame.K_BACKQUOTE,
            "toggle_editor": pygame.K_F1,
            "undo": pygame.K_z,  # With CTRL (handled in DionENG.handle_input)
            "redo": pygame.K_y,  # With CTRL
            "select_tool_move": pygame.K_g,  # Blender-style editor tools
            "select_tool_rotate": pygame.K_r,
            "select_tool_scale": pygame.K_s,
            # Multiplayer (FPS, MMO)
            "open_chat": pygame.K_t,
            "team_chat": pygame.K_y,
            "voice_chat": pygame.K_v,
            "switch_team": pygame.K_m,
            # Modding/Debug
            "toggle_debug": pygame.K_F3,
            "hot_reload_script": pygame.K_F4,
            "toggle_profiler": pygame.K_F6,
            # RTS/Strategy
            "build_structure": pygame.K_b,
            "select_unit": pygame.K_q,
            "group_unit_1": pygame.K_1,
            "group_unit_2": pygame.K_2,
            "group_unit_3": pygame.K_3,
            "attack_move": pygame.K_a,
            # RPG/Adventure
            "interact": pygame.K_e,
            "use_item": pygame.K_u,
            "open_quest_log": pygame.K_l,
            # Platformer
            "dash": pygame.K_x,
            "wall_jump": pygame.K_z,
            # General
            "pause": pygame.K_p,
            "quit": pygame.K_F12
        }

    def handle_input(self, event, callback):
        if event.type == pygame.KEYDOWN:
            for action, key in self.keys.items():
                if event.key == key:
                    callback(action)