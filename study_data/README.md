# Use the NPZ data

Run these commands from the repository's main folder:

```bash
python train.py --data-dir study_data --dataset culex --split all --task classification --model ann
python train.py --data-dir study_data --dataset culex --split all --task regression --model ann
```

Replace `culex` with `thai-sonication`, `thai-bead`, or `lakewood`.

See the [usage instructions](../README.md) for more options.
