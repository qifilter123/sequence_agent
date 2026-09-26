# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from pathlib import Path


LOOKBACK_BARS = 26

UP_RETURN_THRESHOLD = 0.003
DOWN_RETURN_THRESHOLD = -0.003
CLASS_NAMES = ("UP", "FLAT", "DOWN")
UP_CLASS = 0
FLAT_CLASS = 1
DOWN_CLASS = 2


PRICE_FEATURE_INDEX = 0
VOLUME_FEATURE_INDEX = 1
INPUT_DIM = 2
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 64
DROPOUT = 0.1
POSITIONAL_BASE = 10_000.0

VALIDATION_FRACTION = 0.2
VALIDATION_GAP_SAMPLES = LOOKBACK_BARS
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
SHUFFLE_TRAINING = True
NUM_WORKERS = 0
RANDOM_SEED = 7
DEVICE = "auto"
EPSILON = 1e-12
MODEL_OUTPUT_PATH = Path(__file__).with_name("transformer.pt")
