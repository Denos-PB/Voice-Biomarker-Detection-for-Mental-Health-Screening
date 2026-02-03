from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

@dataclass
class Recording:
    id: str
    path: Path
    emotion: Optional[str] = None
    actor: Optional[str] = None
    phq9_total: Optional[int] = None
    gad7_total: Optional[int] = None

def load_metadata(csv_path: Path, audio_root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "id" not in df.columns or "audio_filename" not in df.columns:
        raise ValueError("metadata.csv must contain 'id' and 'audio_filename'.")

    df["audio_path"] = df["audio_filename"].apply(lambda name: str(audio_root / name))
    return df

def dataframe_to_recordings(df: pd.DataFrame) -> list[Recording]:
    recs: list[Recording] = []

    has_phq9 = "phq9_total" in df.columns
    has_gad7 = "gad7_total" in df.columns

    for _, row in df.iterrows():
        phq9_val = row.at["phq9_total"] if has_phq9 else None
        gad7_val = row.at["gad7_total"] if has_gad7 else None

        phq9_total = int(phq9_val) if phq9_val is not None and not pd.isna(phq9_val) else None
        gad7_total = int(gad7_val) if gad7_val is not None and not pd.isna(gad7_val) else None

        recs.append(
            Recording(
                id=str(row["id"]),
                path=Path(str(row["audio_path"])),
                emotion=row.get("emotion"),
                actor=str(row.get("actor")) if "actor" in row else None,
                phq9_total=phq9_total,
                gad7_total=gad7_total,
            )
        )

    return recs