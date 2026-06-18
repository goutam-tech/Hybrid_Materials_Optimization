from matminer.datasets import load_dataset
import pandas as pd

for name in ['matbench_mp_gap', 'matbench_log_kvrh', 'matbench_log_gvrh']:
    df = load_dataset(name)
    df.to_json(f'datasets/{name}.json.gz', orient='records', compression='gzip')