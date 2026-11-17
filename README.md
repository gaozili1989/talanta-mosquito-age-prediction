# How to use

## Install

```bash
git clone https://github.com/gaozili1989/talanta-mosquito-age-prediction.git
cd talanta-mosquito-age-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional demo:

```bash
python quickstart.py
```

## Run classification or regression

```bash
python train.py --data-dir study_data --dataset culex --split all --task classification --model ann
python train.py --data-dir study_data --dataset culex --split all --task regression --model ann
```

Replace `culex` with `thai-sonication`, `thai-bead`, or `lakewood`. Classification and regression data are in NPZ format.

- Splits: `--split 1`, `--split 1 3`, or `--split all`.
- Classification models: `ann`, `svm`, `logistic`, or `all`.
- Regression models: `ann`, `svr`, `pls`, or `all`.

## Choose preprocessing and training settings

Check the available wavenumber range:

```bash
python train.py --data-dir study_data --dataset culex --show-range
```

Add any of these options to a training command:

| Option | Use |
| --- | --- |
| `--range 650 1850` | Select a range within the available wavenumbers (cm⁻¹). |
| `--full-range` | Use all available wavenumbers. |
| `--normalize` / `--no-normalize` | Turn normalization on or off. |
| `--second-derivative` / `--no-second-derivative` | Turn the second derivative on or off. |
| `--pca-components 100` | Use 100 PCA components; omit for no PCA. |
| `--epochs 10000` | Set ANN training epochs. |
| `--learning-rate 0.001` | Set the ANN learning rate. |
| `--device cpu` / `--device cuda` | Choose CPU or NVIDIA GPU. |

Defaults: range 650–1850 cm⁻¹ for the supplied datasets, normalization on, second derivative off, PCA off, 10,000 epochs, learning rate 0.001.

## Run in Google Colab

```python
!git clone https://github.com/gaozili1989/talanta-mosquito-age-prediction.git
%cd /content/talanta-mosquito-age-prediction
%pip install -r requirements.txt
!python train.py --data-dir study_data --dataset culex --split all --task classification --model ann
```

Change `--task classification` to `--task regression` for regression.

## Find results

Results are saved under `runs/`: predictions, accuracy, MAE, RMSE, Pearson R, and figures appropriate to each task. Multiple runs also produce `summary.csv`.

Use `--output-dir runs/my_run` to choose a new output folder. Add `--no-tsne` to skip t-SNE or `--no-diagnostics` to skip diagnostic figures.

For all available options:

```bash
python train.py --help
```
