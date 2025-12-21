from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    ALTITUDE_KM: float = 600
    OBSERVATION_ANGLE_MAX_DEG: float = 90
    THERMAL_SUN_ANGLE_RANGE_DEG: tuple[float, float] = (45, 135)
    THERMAL_Az_MAX_DEG: float = 90
    THERMAL_Zn_MAX_DEG: float = 24

    TARGET_CATALOG_PATH: Path | None = None
    OUTPUT_DIR: Path | None = None

DEFAULT_CONFIG = Config()
