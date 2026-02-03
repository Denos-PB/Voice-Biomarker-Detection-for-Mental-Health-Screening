import os
from typing import Dict, Any, List, cast
import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

class AudioAugmentation:
    """
    Create realistic noisy / degraded versions of clean speech
    recordings (e.g., RAVDESS) for robustness testing.

    This class focuses on *acoustic channel* and *environment*
    perturbations, not semantic content changes.
    """

    def __init__(self, audio_path: str, sr: int = 22_050, random_state: int | None = None) -> None:
        if random_state is not None:
            np.random.seed(random_state)

        self.audio, self.sr = librosa.load(audio_path, sr=sr, mono=True)
        self.duration = librosa.get_duration(y=self.audio, sr=sr)

    def add_background_noise(self, noise_type: str = "cafe", snr_db: float = 10.0) -> tuple[np.ndarray, Dict[str, Any]]:
        noise_generators = {
            "cafe": self._generate_cafe_noise,
            "office": self._generate_office_noise,
            "street": self._generate_street_noise,
            "subway": self._generate_subway_noise,
            "wind": self._generate_wind_noise,
            "rain": self._generate_rain_noise,
        }

        if noise_type not in noise_generators:
            raise ValueError(f"Unsupported noise_type: {noise_type}")

        noise = noise_generators[noise_type]()

        if len(noise) < len(self.audio):
            noise = np.tile(noise, int(np.ceil(len(self.audio) / len(noise))))
        noise = noise[: len(self.audio)]

        signal_power = np.mean(self.audio**2)
        noise_power = np.mean(noise**2) + 1e-12

        scaling_factor = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
        noisy_audio = self.audio + scaling_factor * noise

        return noisy_audio, {
            "noise_type": noise_type,
            "snr_db": snr_db,
            "actual_snr": float(self._calculate_snr(self.audio, noisy_audio)),
        }

    def _generate_cafe_noise(self) -> np.ndarray:
        duration = self.duration
        sr = self.sr

        babble = np.zeros(int(duration * sr))
        t = np.linspace(0, duration, len(babble))

        for _ in range(np.random.randint(5, 10)):  
            freq = np.random.uniform(100, 300)
            phase = np.random.uniform(0, 2 * np.pi)
            babble += np.sin(2 * np.pi * freq * t + phase) * np.random.uniform(0.1, 0.3)

        clinks = np.zeros_like(babble)
        n_clinks = np.random.randint(10, 30)
        clink_positions = np.random.choice(len(clinks), n_clinks, replace=False)
        clinks[clink_positions] = np.random.uniform(0.5, 1.0, n_clinks)

        freqs = [50, 100, 150, 200]
        music = sum(
            np.sin(2 * np.pi * f * np.linspace(0, duration, len(babble))) for f in freqs
        ) * 0.1

        return babble + clinks + music

    def _generate_office_noise(self) -> np.ndarray:
        duration = self.duration
        sr = self.sr

        t = np.linspace(0, duration, int(duration * sr))
        hum = 0.05 * np.sin(2 * np.pi * 60 * t)

        typing = np.zeros_like(hum)
        n_keystrokes = int(duration * np.random.uniform(2, 5))
        keystroke_positions = np.random.choice(len(typing), n_keystrokes, replace=False)
        for pos in keystroke_positions:
            click_duration = int(0.05 * sr)
            if pos + click_duration < len(typing):
                window = np.asarray(signal.get_window("hann", click_duration))
                typing[pos : pos + click_duration] = (
                    np.random.uniform(0.2, 0.4) * window
                )

        pink_noise = self._generate_pink_noise(len(hum))
        rustling = pink_noise * 0.03

        return hum + typing + rustling

    def _generate_street_noise(self) -> np.ndarray:
        t = np.linspace(0, self.duration, int(self.duration * self.sr))

        car_freqs = [80, 120, 160, 200]
        cars = sum(
            np.sin(2 * np.pi * f * t) * np.random.uniform(0.1, 0.3) for f in car_freqs
        )

        wind = self._generate_wind_noise()
        voices = self._generate_pink_noise(len(t)) * 0.15

        return cars + wind + voices

    def _generate_subway_noise(self) -> np.ndarray:
        t = np.linspace(0, self.duration, int(self.duration * self.sr))

        rumble = np.zeros_like(t, dtype=float)
        for f, a in zip([30, 60, 90, 120], [0.5, 0.4, 0.3, 0.2]):
            rumble += np.sin(2 * np.pi * f * t) * a

        screech: Any = np.random.randn(len(t)) * 0.2
        screech = signal.sosfilt(signal.butter(4, 2000, "high", fs=self.sr, output="sos"), screech)
        if isinstance(screech, tuple):
            screech = screech[0]
        screech = np.asarray(screech)

        screech *= (np.random.rand(len(screech)) > 0.97).astype(float) * 2

        return cast(np.ndarray, np.asarray(rumble + screech))

    def _generate_wind_noise(self) -> np.ndarray:
        white: Any = np.random.randn(int(self.duration * self.sr))
        sos = signal.butter(4, 200, "low", fs=self.sr, output="sos")
        wind: Any = signal.sosfilt(sos, white)
        if isinstance(wind, tuple):
            wind = wind[0]
        return cast(np.ndarray, np.asarray(wind) * 0.1)

    def _generate_rain_noise(self) -> np.ndarray:
        white: Any = np.random.randn(int(self.duration * self.sr))
        sos = signal.butter(4, [1000, 8000], "band", fs=self.sr, output="sos")
        rain: Any = signal.sosfilt(sos, white)
        if isinstance(rain, tuple):
            rain = rain[0]
        return cast(np.ndarray, np.asarray(rain) * 0.15)

    def _generate_pink_noise(self, length: int) -> np.ndarray:
        white = np.random.randn(length)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.5221894]
        pink = signal.lfilter(b, a, white)
        pink /= np.max(np.abs(pink)) + 1e-12
        return pink

    def add_microphone_distortion(self, quality: str = "medium") -> np.ndarray:
        audio: Any = self.audio.copy()

        if quality == "professional":

            return cast(np.ndarray, audio)

        if quality == "medium":
            sos = signal.butter(2, [100, 8000], "band", fs=self.sr, output="sos")
            audio = signal.sosfilt(sos, audio)

        elif quality == "phone":
            sos = signal.butter(4, [300, 3400], "band", fs=self.sr, output="sos")
            audio = signal.sosfilt(sos, audio)
            audio = np.clip(audio, -0.8, 0.8)

        elif quality == "cheap_headset":
            sos_high = signal.butter(2, 2000, "high", fs=self.sr, output="sos")
            audio = signal.sosfilt(sos_high, audio)
            hiss = np.random.randn(len(self.audio)) * 0.01
            audio = audio + hiss

        else:
            raise ValueError(f"Unsupported quality: {quality}")

        if isinstance(audio, tuple):
            audio = audio[0]
        return cast(np.ndarray, np.asarray(audio, dtype=self.audio.dtype))

    def add_compression_artifacts(self, codec: str = "mp3", bitrate: int = 64) -> np.ndarray:
        stft = librosa.stft(self.audio)

        if codec == "mp3":
            if bitrate <= 64:
                stft[int(len(stft) * 0.7) :] *= 0.3
            elif bitrate <= 128:
                stft[int(len(stft) * 0.85) :] *= 0.6

        elif codec == "phone":
            stft[: int(len(stft) * 0.1)] *= 0.2
            stft[int(len(stft) * 0.4) :] *= 0.1

        else:
            raise ValueError(f"Unsupported codec: {codec}")

        compressed: Any = librosa.istft(stft)
        if isinstance(compressed, tuple):
            compressed = compressed[0]
        arr = np.asarray(compressed[: len(self.audio)], dtype=self.audio.dtype)
        return cast(np.ndarray, arr)

    def add_clipping(self, threshold: float = 0.8) -> np.ndarray:
        return np.clip(self.audio, -threshold, threshold)

    def add_dropouts(self, n_dropouts: int = 3, dropout_duration: float = 0.1) -> np.ndarray:
        audio = self.audio.copy()
        drop_len = int(dropout_duration * self.sr)

        for _ in range(n_dropouts):
            if drop_len >= len(audio):
                break
            start = np.random.randint(0, len(audio) - drop_len)
            end = start + drop_len

            fade_len = int(0.01 * self.sr)
            fade_len = min(fade_len, drop_len // 2)
            audio[start : start + fade_len] *= np.linspace(1, 0, fade_len)
            audio[start + fade_len : end - fade_len] = 0.0
            audio[end - fade_len : end] *= np.linspace(0, 1, fade_len)

        return audio

    def add_room_reverb(self, room_size: str = "medium", dampening: float = 0.5) -> np.ndarray:
        if room_size == "small":
            reverb_time = 0.3
            early_reflections = 5
        elif room_size == "medium":
            reverb_time = 0.8
            early_reflections = 10
        elif room_size == "large":
            reverb_time = 1.5
            early_reflections = 15
        else:
            reverb_time = 3.0
            early_reflections = 25

        ir_length = int(reverb_time * self.sr)
        ir = np.zeros(ir_length)

        early_time = int(0.05 * self.sr)
        for i in range(early_reflections):
            delay = np.random.randint(int(0.01 * self.sr), early_time)
            amplitude = np.random.uniform(0.3, 0.7) * (1 - i / early_reflections)
            if delay < len(ir):
                ir[delay] += amplitude

        decay = np.exp(-6 * np.arange(ir_length) / (reverb_time * self.sr))
        late_reverb = np.random.randn(ir_length) * decay * dampening * 0.1
        ir += late_reverb

        reverberant = signal.convolve(self.audio, ir, mode="same")
        mixed = 0.7 * self.audio + 0.3 * reverberant

        mixed /= np.max(np.abs(mixed)) + 1e-12
        return mixed

    def add_distance_effect(self, distance_meters: float = 5.0) -> np.ndarray:
        audio = self.audio.copy()

        attenuation = 1.0 / max(distance_meters, 1.0)
        audio *= attenuation

        cutoff = max(1000, 8000 - distance_meters * 500)
        sos = signal.butter(2, cutoff, "low", fs=self.sr, output="sos")
        audio = signal.sosfilt(sos, audio)
        if isinstance(audio, tuple):
            audio = audio[0]
        audio = np.asarray(audio, dtype=self.audio.dtype)

        if distance_meters > 2:
            room_size = "medium" if distance_meters < 10 else "large"
            tmp = AudioAugmentation.__new__(AudioAugmentation)
            tmp.audio = audio
            tmp.sr = self.sr
            tmp.duration = self.duration
            audio = AudioAugmentation.add_room_reverb(tmp, room_size=room_size)

        return audio

    def denoise(self, stationary: bool = False) -> np.ndarray:
        if stationary:
            raw_result: Any = nr.reduce_noise(y=self.audio, sr=self.sr, stationary=True)
        else:
            raw_result = nr.reduce_noise(y=self.audio, sr=self.sr)

        if isinstance(raw_result, tuple):
            raw_result = raw_result[0]

        arr = np.asarray(raw_result, dtype=self.audio.dtype)
        return cast(np.ndarray, arr)

    def _calculate_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        noise = noisy - clean
        signal_power = np.mean(clean**2) + 1e-12
        noise_power = np.mean(noise**2) + 1e-12
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    def save_augmented_dataset(self, output_dir: str, augmentation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)

        metadata: List[Dict[str, Any]] = []

        if "background_noise" in augmentation_config:
            cfg = augmentation_config["background_noise"]
            for noise_type in cfg.get("types", []):
                for snr in cfg.get("snr_levels", []):
                    noisy, info = self.add_background_noise(noise_type, snr)
                    filename = f"bg_{noise_type}_snr{snr}dB.wav"
                    filepath = os.path.join(output_dir, filename)
                    sf.write(filepath, noisy, self.sr)
                    metadata.append(
                        {
                            "filename": filename,
                            "augmentation": "background_noise",
                            **info,
                        }
                    )

        if "microphone" in augmentation_config:
            cfg = augmentation_config["microphone"]
            for quality in cfg.get("qualities", []):
                aug_audio = self.add_microphone_distortion(quality=quality)
                filename = f"mic_{quality}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, aug_audio, self.sr)
                metadata.append(
                    {
                        "filename": filename,
                        "augmentation": "microphone",
                        "quality": quality,
                    }
                )

        if "distance" in augmentation_config:
            cfg = augmentation_config["distance"]
            for dist in cfg.get("distances", []):
                aug_audio = self.add_distance_effect(distance_meters=dist)
                filename = f"distance_{dist}m.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, aug_audio, self.sr)
                metadata.append(
                    {
                        "filename": filename,
                        "augmentation": "distance",
                        "distance_meters": dist,
                    }
                )

        if "compression" in augmentation_config:
            cfg = augmentation_config["compression"]
            for codec in cfg.get("codecs", []):
                for br in cfg.get("bitrates", []):
                    aug_audio = self.add_compression_artifacts(codec=codec, bitrate=br)
                    filename = f"codec_{codec}_{br}kbps.wav"
                    filepath = os.path.join(output_dir, filename)
                    sf.write(filepath, aug_audio, self.sr)
                    metadata.append(
                        {
                            "filename": filename,
                            "augmentation": "compression",
                            "codec": codec,
                            "bitrate_kbps": br,
                        }
                    )

        if "combined" in augmentation_config:
            cfg = augmentation_config["combined"]
            for i, scenario in enumerate(cfg.get("scenarios", [])):
                audio = self.audio.copy()

                if "noise" in scenario and "snr" in scenario:
                    noise_audio, _ = self.add_background_noise(
                        noise_type=scenario["noise"],
                        snr_db=scenario["snr"],
                    )
                    audio = noise_audio

                if "mic" in scenario:
                    tmp = AudioAugmentation.__new__(AudioAugmentation)
                    tmp.audio = audio
                    tmp.sr = self.sr
                    tmp.duration = self.duration
                    audio = AudioAugmentation.add_microphone_distortion(tmp, quality=scenario["mic"])

                if "distance" in scenario:
                    tmp = AudioAugmentation.__new__(AudioAugmentation)
                    tmp.audio = audio
                    tmp.sr = self.sr
                    tmp.duration = self.duration
                    audio = AudioAugmentation.add_distance_effect(
                        tmp, distance_meters=scenario["distance"]
                    )

                if "codec" in scenario:
                    tmp = AudioAugmentation.__new__(AudioAugmentation)
                    tmp.audio = audio
                    tmp.sr = self.sr
                    tmp.duration = self.duration
                    audio = AudioAugmentation.add_compression_artifacts(
                        tmp,
                        codec=scenario.get("codec", "mp3"),
                        bitrate=scenario.get("bitrate", 64),
                    )

                filename = f"combined_scenario_{i}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, audio, self.sr)
                metadata.append(
                    {
                        "filename": filename,
                        "augmentation": "combined",
                        **scenario,
                    }
                )

        meta_path = os.path.join(output_dir, "augmentation_metadata.csv")
        pd.DataFrame(metadata).to_csv(meta_path, index=False)

        return metadata


def create_challenge_dataset() -> None:
    ravdess_path = "path/to/ravdess/audio.wav"
    aug = AudioAugmentation(ravdess_path)

    augmentation_config = {
        "background_noise": {
            "types": ["cafe", "office", "street", "subway"],
            "snr_levels": [20, 15, 10, 5],
        },
        "microphone": {
            "qualities": ["medium", "phone", "cheap_headset"],
        },
        "distance": {
            "distances": [2, 5, 10, 15],
        },
        "compression": {
            "codecs": ["mp3"],
            "bitrates": [64, 128],
        },
        "combined": {
            "scenarios": [
                {"noise": "subway", "snr": 5, "mic": "phone", "codec": "phone"},
                {"noise": "cafe", "snr": 10, "mic": "cheap_headset"},
                {"noise": "street", "snr": 8, "distance": 10},
            ]
        },
    }

    aug.save_augmented_dataset("ravdess_augmented/", augmentation_config)