# Annual-Project-ESGI

## Objective
The goal is to implement machine learning models in Rust.

We work on a game screenshot classification task with 3 classes:
- FPS
- METROIDVANIA
- MOBA


## Technologies
- Rust
- Python
- ctypes
- numpy
- Pillow

## Project structure
- `lib_classification/src/linear.rs`: linear model used for the professor-style case
- `lib_classification/src/linear_dataset.rs`: linear model adapted for the dataset
- `lib_classification/src/MLP.rs`: naive MLP core
- `lib_classification/src/mlp_ffi.rs`: FFI wrappers for the MLP
- `PythonProject/rust_bridge.py`: Python/Rust bridge
- `PythonProject/dataset_loader.py`: dataset loading and preprocessing
- `PythonProject/run_dataset.py`: dataset training and evaluation script

## Build
```powershell
cd .\lib_classification
cargo build
```

## Test on Dataset
Before launching the dataset test, update the dataset path so it matches the path on your machine.

In `PythonProject/dataset_loader.py`, change `DEFAULT_DATASET_ROOT` so it points to your local dataset folder.


```powershell
cd .\PythonProject
```

Launch the dataset script with:
```powershell
py .\run_dataset.py
```

If needed, you can also test other preprocessing settings:
```powershell
py .\run_dataset.py --grayscale
py .\run_dataset.py --rgb
py .\run_dataset.py --width 16 --height 16
```