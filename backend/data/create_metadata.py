from pathlib import Path
import csv

AUDIO_ROOT = Path("data/raw/ravdess/audio")
METADATA_PATH = AUDIO_ROOT.parent / "metadata.csv"

MODALITY_MAP = {
    "01": "full_av",
    "02": "video_only",
    "03": "audio_only",
}

CHANNEL_MAP = {
    "01": "speech",
    "02": "song",
}

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}

STATEMENT_MAP = {
    "01": "kids_talking_by_the_door",
    "02": "dogs_are_sitting_by_the_door",
}

REPETITION_MAP = {
    "01": 1,
    "02": 2,
}

def main() -> None:
    audio_root = Path("backend/data/raw/ravdess/audio").resolve()
    metadata_path = audio_root.parent / "metadata.csv"

    rows = []

    wav_files = list[Path](AUDIO_ROOT.rglob("*.wav"))
    all_files = sorted(wav_files)

    if not all_files:
        raise FileNotFoundError(f"No audio files found under {AUDIO_ROOT}")

    for path in all_files:
        fname = path.name
        stem = path.stem
        parts = stem.split("-")

        if len(parts) != 7:
            print(f"Skipping unexpected filename: {fname}")
            continue

        modality_code, channel_code, emotion_code, intensity_code, \
            statement_code, repetition_code, actor_code = parts

        modality = MODALITY_MAP.get(modality_code, "unknown")
        channel = CHANNEL_MAP.get(channel_code, "unknown")
        emotion = EMOTION_MAP.get(emotion_code, "unknown")
        intensity = INTENSITY_MAP.get(intensity_code, "unknown")
        statement = STATEMENT_MAP.get(statement_code, "unknown")
        repetition = REPETITION_MAP.get(repetition_code, None)
        actor = int(actor_code)
        gender = "male" if actor % 2 == 1 else "female"

        row = {
            "id": stem,
            "audio_filename": fname,
            "modality": modality,
            "vocal_channel": channel,
            "emotion": emotion,
            "emotional_intensity": intensity,
            "statement": statement,
            "repetition": repetition,
            "actor": actor,
            "actor_gender": gender,
            "phq9_total": "",
            "gad7_total": "",
        }
        rows.append(row)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "audio_filename",
        "modality",
        "vocal_channel",
        "emotion",
        "emotional_intensity",
        "statement",
        "repetition",
        "actor",
        "actor_gender",
        "phq9_total",
        "gad7_total",
    ]

    with METADATA_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {METADATA_PATH}")

if __name__ == "__main__":
    main()