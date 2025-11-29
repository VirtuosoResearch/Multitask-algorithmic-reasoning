## Generating datasets
```python
from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=10000, graph_generator="er", graph_generator_kwargs={"n": [16, 32], "p_range": (0.1, 0.3)}, hints=True)
```

## Run code
```bash
python baselines/run_experiment.py --cfg [baselines/configs/bfs/GIN.yml] --seed 42 --data-dir [data/] --hints --enable-wandb --size [5000] --node [16] --algorithm [bfs]
```