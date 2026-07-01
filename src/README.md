# AuraXR Python Modules

This directory implements the model/data contracts described in `docs/`.

- `dataset_oakink.py`: OakInk static grasp loader.
- `dataset_hot3d.py`: HOT3D temporal sliding-window loader.
- `grasp_model.py`: single Temporal Geometry-Conditioned Grasp Model.
- `train_grasp.py`: static, temporal, or mixed training.
- `evaluate_contracts.py`: validates dataset/model/Unity I/O shapes.
- `export_onnx.py`: exports static or temporal ONNX interface.
- `unity_contract.py`: JSON schema helper for Unity eval logs.

Quick checks:

```bash
python3 src/inspect_processed_data.py
python3 src/evaluate_contracts.py --batch 2 --window 16
python3 src/train_grasp.py --mode static --epochs 1 --steps_per_epoch 2 --batch 4
```

