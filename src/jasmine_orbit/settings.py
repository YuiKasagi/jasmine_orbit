from pathlib import Path

# ---- 定数 ----
DEFAULT_ALTITUDE_KM = 600

OBSERVATION_ANGLE_MAX_DEG = 90
THERMAL_SUN_ANGLE_RANGE_DEG = (45, 135)
THERMAL_Az_MAX_DEG = 90
THERMAL_Zn_MAX_DEG = 24

TARGET_CATALOG_PATH = Path("./data/exoplanet_target.csv")
OUTPUT_DIR = Path("./output/")
# ----------------