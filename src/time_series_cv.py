import numpy as np

def expanding_window_splits(n, initial, step=1, horizon=1, max_splits=None):
    """
    Yield (train_idx, test_idx) for expanding window.
    initial: jumlah awal data untuk train
    step: berapa banyak maju setiap split
    horizon: berapa banyak titik prediksi (biasanya 1 untuk 1-step)
    """
    splits = 0
    for end_train in range(initial, n - horizon + 1, step):
        train_idx = np.arange(0, end_train)
        test_idx = np.arange(end_train, end_train + horizon)
        yield train_idx, test_idx
        splits += 1
        if max_splits and splits >= max_splits:
            break
