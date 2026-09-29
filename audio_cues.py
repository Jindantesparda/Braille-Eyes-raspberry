"""Low-cost, precomputed stereo tone cues for calibrated obstacle depths."""

import numpy as np


class AudioCuePlayer:
    SAMPLE_RATE = 22_050
    DURATION_SECONDS = 0.12

    def __init__(self):
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("install pygame to enable audio cues") from error

        self.pygame = pygame
        try:
            pygame.mixer.pre_init(
                frequency=self.SAMPLE_RATE,
                size=-16,
                channels=2,
                buffer=512,
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(1)
        except pygame.error as error:
            raise RuntimeError(f"could not initialize audio output: {error}") from error

        self.channel = pygame.mixer.Channel(0)
        self.sounds = self._make_sounds()

    def _make_sounds(self):
        # Cache every short cue once. The navigation loop only chooses and plays it.
        frames = int(self.SAMPLE_RATE * self.DURATION_SECONDS)
        times = np.arange(frames, dtype=np.float32) / self.SAMPLE_RATE
        fade_frames = max(1, int(self.SAMPLE_RATE * 0.008))
        envelope = np.ones(frames, dtype=np.float32)
        envelope[:fade_frames] = np.linspace(0.0, 1.0, fade_frames)
        envelope[-fade_frames:] = np.linspace(1.0, 0.0, fade_frames)

        bands = {
            "near": (880, 0.48),
            "medium": (660, 0.34),
            "far": (440, 0.22),
        }
        pans = {
            "left": (1.0, 0.22),
            "center": (0.72, 0.72),
            "right": (0.22, 1.0),
        }
        sounds = {}
        for band, (frequency, volume) in bands.items():
            tone = np.sin(2.0 * np.pi * frequency * times) * envelope
            for direction, (left_gain, right_gain) in pans.items():
                stereo = np.column_stack((
                    tone * volume * left_gain,
                    tone * volume * right_gain,
                ))
                pcm = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
                sounds[(band, direction)] = self.pygame.mixer.Sound(
                    buffer=pcm.tobytes()
                )
        return sounds

    @staticmethod
    def _band_for_distance(distance_m):
        if distance_m <= 1.0:
            return "near"
        if distance_m <= 2.5:
            return "medium"
        return "far"

    def play(self, distance_m, direction):
        if direction not in ("left", "center", "right"):
            return
        self.channel.play(
            self.sounds[(self._band_for_distance(distance_m), direction)]
        )

    def close(self):
        self.channel.stop()
        self.pygame.mixer.quit()
