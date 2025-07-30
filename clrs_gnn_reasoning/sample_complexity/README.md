## Generating datasets
```python
from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=10000, graph_generator="er", graph_generator_kwargs={"n": [16, 32], "p_range": (0.1, 0.3)}, hints=True)
```

## Run code
```bash
python baselines/run_experiment.py --cfg baselines/configs/dijkstra/GINE.yml --seed 42 --data-dir path/to/data/store --enable-wandb --hints
```