import os
import numpy as np


WINDOW = 1024
STRIDE = 256
TRAIN_LEN = 4000
TOTAL_WINDOWS = 7997
TEST_TOTAL = 8000
NORMAL_REST = TOTAL_WINDOWS - TRAIN_LEN  # 3997
ABNORMAL_FRONT = TEST_TOTAL - NORMAL_REST  # 4003
SEG_MIN = 50
SEG_MAX = 100


def load_series(txt_path: str) -> np.ndarray:
    with open(txt_path, "r") as f:
        data = [float(line.strip()) for line in f if line.strip()]
    arr = np.asarray(data, dtype=np.float64)
    return arr


def make_windows(series: np.ndarray, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    n = (len(series) - window) // stride + 1
    windows = np.array([series[i:i + window] for i in range(0, n * stride, stride)], dtype=np.float64)
    return windows


def split_into_segments(total: int, rng: np.random.Generator,
                        seg_min: int = SEG_MIN, seg_max: int = SEG_MAX) -> list:
    lengths = []
    remaining = total
    # Keep remaining > 200 to guarantee two-segment finish
    while remaining > 200:
        l = int(rng.integers(seg_min, seg_max + 1))
        lengths.append(l)
        remaining -= l
    # Finish with 2 or 1 segments such that each in [seg_min, seg_max]
    if remaining > seg_max:
        # remaining in (100, 200]
        low = seg_min
        high = min(seg_max, remaining - seg_min)
        l = int(rng.integers(low, high + 1))
        lengths.append(l)
        remaining -= l
        # remaining now in [seg_min, seg_max]
        lengths.append(remaining)
        remaining = 0
    else:
        # remaining in [seg_min, seg_max]
        lengths.append(remaining)
        remaining = 0
    assert sum(lengths) == total, f"Segment sum {sum(lengths)} != total {total}"
    assert all(SEG_MIN <= l <= SEG_MAX for l in lengths), "Segment out of bounds"
    return lengths


def build_selfbuilt_dataset(base_dir: str, seed: int = 42):
    rng = np.random.default_rng(seed)

    normal_path = os.path.join(base_dir, "bulou.txt")
    abnormal_path = os.path.join(base_dir, "lou.txt")

    normal_series = load_series(normal_path)
    abnormal_series = load_series(abnormal_path)

    assert len(normal_series) == 2_048_000, f"Normal length {len(normal_series)} != 2048000"
    assert len(abnormal_series) == 2_048_000, f"Abnormal length {len(abnormal_series)} != 2048000"

    normal_windows = make_windows(normal_series)
    abnormal_windows = make_windows(abnormal_series)

    assert normal_windows.shape == (TOTAL_WINDOWS, WINDOW)
    assert abnormal_windows.shape == (TOTAL_WINDOWS, WINDOW)

    # Train: first 4000 normal windows
    train = normal_windows[:TRAIN_LEN]

    # Test: interleave remaining normal (3997) and first 4003 abnormal windows by segments 50-100
    normal_rest = normal_windows[TRAIN_LEN:TRAIN_LEN + NORMAL_REST]
    abnormal_front = abnormal_windows[:ABNORMAL_FRONT]

    n_segs = split_into_segments(NORMAL_REST, rng)
    a_segs = split_into_segments(ABNORMAL_FRONT, rng)

    # Interleave segments starting with normal
    test_chunks = []
    labels = []
    i_n = i_a = 0
    pos_n = pos_a = 0
    turn_normal = True

    while i_n < len(n_segs) or i_a < len(a_segs):
        if turn_normal:
            if i_n < len(n_segs):
                l = n_segs[i_n]
                test_chunks.append(normal_rest[pos_n:pos_n + l])
                labels.append(np.zeros(l, dtype=np.int64))
                pos_n += l
                i_n += 1
            # flip turn regardless to ensure alternation
            turn_normal = False
        else:
            if i_a < len(a_segs):
                l = a_segs[i_a]
                test_chunks.append(abnormal_front[pos_a:pos_a + l])
                labels.append(np.ones(l, dtype=np.int64))
                pos_a += l
                i_a += 1
            turn_normal = True

    test = np.vstack(test_chunks)
    labels_arr = np.concatenate(labels)

    assert test.shape == (TEST_TOTAL, WINDOW), f"Test shape {test.shape} != (8000,1024)"
    assert labels_arr.shape == (TEST_TOTAL,), f"Labels shape {labels_arr.shape} != (8000,)"
    assert labels_arr.sum() == ABNORMAL_FRONT, "Label count of abnormal should equal 4003"

    out_dir = os.path.join(base_dir, "dataset", "SelfBuilt")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train.npy"), train)
    np.save(os.path.join(out_dir, "test.npy"), test)
    np.save(os.path.join(out_dir, "labels.npy"), labels_arr)

    print("Generated files:")
    print(f"  train.npy: {train.shape}")
    print(f"  test.npy:  {test.shape}")
    print(f"  labels.npy:{labels_arr.shape}, abnormal_count={labels_arr.sum()}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    build_selfbuilt_dataset(BASE_DIR, seed=42)