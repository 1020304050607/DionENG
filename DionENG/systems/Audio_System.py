import pygame
from .. import Component, Transform

class AudioSystem(Component):
    def __init__(self):
        super().__init__()
        self.sounds = {}  # Stores pygame.mixer.Sound objects
        self.music = None  # Current background music
        self.channels = {}  # Maps sound_name to pygame.mixer.Channel
        self.sound_categories = {
            "sfx": [],  # Sound effects (e.g., gunshots, UI clicks)
            "music": [],  # Background music tracks
            "ambient": [],  # Looping environmental sounds
            "voice": []  # Dialogue or voice chat
        }
        self.max_channels = 32  # Limit for concurrent sounds
        pygame.mixer.set_num_channels(self.max_channels)
        self.listener_position = (0, 0, 0)  # For 3D audio

    def load_sound(self, sound_name, category="sfx"):
        """Load a sound into self.sounds and assign it to a category."""
        try:
            if sound_name not in self.sounds:
                sound = self.sounds.get(sound_name, pygame.mixer.Sound(f"assets/{sound_name}"))
                self.sounds[sound_name] = sound
                self.sound_categories[category].append(sound_name)
                print(f"Loaded sound: {sound_name} in category {category}")
        except Exception as e:
            print(f"Failed to load sound {sound_name}: {e}")

    def play_sound(self, sound_name, position=None, volume=1.0, pitch=1.0, loop=False):
        """Play a sound with optional 3D positioning, volume, pitch, and looping."""
        if sound_name in self.sounds:
            sound = self.sounds[sound_name]
            # Find an available channel
            channel = pygame.mixer.find_channel()
            if not channel:
                print(f"No available channel for {sound_name}")
                return
            # Apply volume
            channel.set_volume(volume)
            # Simulate 3D audio (basic distance-based volume)
            if position:
                distance = self._calculate_distance(position)
                adjusted_volume = max(0.0, volume * (1.0 - distance / 100.0))  # Attenuate with distance
                channel.set_volume(adjusted_volume)
            # Play sound
            channel.play(sound, loops=-1 if loop else 0)
            self.channels[sound_name] = channel
            # Note: Pygame doesn't support pitch natively; this is a placeholder
            if pitch != 1.0:
                print(f"Pitch adjustment ({pitch}) not supported by pygame.mixer")
            print(f"Playing sound: {sound_name}, volume: {volume}, loop: {loop}")

    def stop_sound(self, sound_name):
        """Stop a specific sound."""
        if sound_name in self.channels:
            self.channels[sound_name].stop()
            del self.channels[sound_name]
            print(f"Stopped sound: {sound_name}")

    def pause_sound(self, sound_name):
        """Pause a specific sound."""
        if sound_name in self.channels:
            self.channels[sound_name].pause()
            print(f"Paused sound: {sound_name}")

    def resume_sound(self, sound_name):
        """Resume a paused sound."""
        if sound_name in self.channels:
            self.channels[sound_name].unpause()
            print(f"Resumed sound: {sound_name}")

    def play_music(self, music_name, volume=0.5, loop=True):
        """Play background music with looping option."""
        try:
            if self.music != music_name:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(f"assets/{music_name}")
                pygame.mixer.music.set_volume(volume)
                pygame.mixer.music.play(-1 if loop else 0)
                self.music = music_name
                self.sound_categories["music"].append(music_name)
                print(f"Playing music: {music_name}, volume: {volume}, loop: {loop}")
        except Exception as e:
            print(f"Failed to play music {music_name}: {e}")

    def stop_music(self):
        """Stop background music."""
        pygame.mixer.music.stop()
        self.music = None
        print("Stopped music")

    def set_listener_position(self, position):
        """Set listener position for 3D audio calculations."""
        self.listener_position = position
        print(f"Listener position set to: {position}")

    def _calculate_distance(self, sound_position):
        """Calculate distance from listener to sound source for 3D audio."""
        if not sound_position:
            return 0.0
        listener = self.listener_position
        return ((sound_position[0] - listener[0])**2 + 
                (sound_position[1] - listener[1])**2 + 
                (sound_position[2] - listener[2])**2)**0.5

    def update(self, entities, assets):
        """Update entity-based audio (e.g., footsteps, ambient sounds)."""
        for entity in entities:
            if hasattr(entity, "audio"):
                audio_data = getattr(entity, "audio")
                if isinstance(audio_data, dict) and "sound_name" in audio_data:
                    transform = entity.get_component("Transform")
                    position = transform.position if transform else None
                    self.play_sound(
                        sound_name=audio_data["sound_name"],
                        position=position,
                        volume=audio_data.get("volume", 1.0),
                        loop=audio_data.get("loop", False)
                    )