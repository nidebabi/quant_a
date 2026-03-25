from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_FEATURES = ROOT / "data" / "features"
DATA_LABELS = ROOT / "data" / "labels"
REPORTS = ROOT / "reports"

for p in [DATA_RAW, DATA_FEATURES, DATA_LABELS, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 数据范围
# ---------------------------
START_DATE = "20220101"

# ---------------------------
# 股票池
# ---------------------------
MAIN_BOARD_PREFIX = ("60", "00")
MIN_LISTING_BARS = 120

# ---------------------------
# 过滤参数
# ---------------------------
MIN_AMOUNT = 2e8
MIN_TURNOVER = 1.0
MAX_TURNOVER = 20.0

RET3_MIN = 0.02
RET3_MAX = 0.18
RET10_MAX = 0.28
CLOSE_LOC_MIN = 0.65
VOL_RATIO_MIN = 1.1
VOL_RATIO_MAX = 2.8
UPPER_SHADOW_MAX = 0.35

# ---------------------------
# 回测参数
# ---------------------------
TOP_N = 20
TAKE_PROFIT = 0.02
STOP_LOSS = -0.01

# 交易成本假设（可自行调）
BUY_FEE_RATE = 0.0003
SELL_FEE_RATE = 0.0008