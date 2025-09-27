# Spectral-based Multiband RNNs for Virtual Analog Compressor Modeling

Implementation of a deep learning framework for virtual analog modeling of dynamic range compressors.   Our method uses STFT magnitude features with spectral amplification and multi-band LSTM/GRU architectures for efficient modeling.  

Includes **datasets, evaluation metrics, and pretrained models** for reproducibility.

- **Pretrain models and other training artifacts:**  https://drive.google.com/file/d/1CgtiMSSyLrxCWJgrQAwCqEp7JEzVtZ24/view?usp=sharing

- **Datasets (Kaggle):** https://www.kaggle.com/datasets/johngolt/vca-compressors-dataset

## Installation
Use the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Data 
Specify dataset paths in ./data/__init__.py:

```python
        'alesis3630': {
            'dataset_class': VCADataset,
            'train_source': "path_to_soruces",
            'train_targets': "path_to_3630",
            'val_source': "path_to_soruces",
            'val_targets': "path_to_3630",
            'test_source': "path_to_soruces",
            'test_targets': "path_to_3630",
            'nparams': 4
        },
```

## Configs
Configs are organized into three folders to study the influence of different network setups:
- STFT features
- Number of LSTM/GRU layers and hidden size
- Number of bands in multi-band setup

Final configs used for comparison:
```bash
# LSTM-based
configs/num_bands_type_abilation/lstm_multiband_1_band.yaml
configs/num_bands_type_abilation/lstm_multiband_4_band.yaml

# GRU-based
configs/num_bands_type_abilation/multiband_gru_1_band.yaml
configs/num_bands_type_abilation/ multiband_gru_4_band.yaml
```

## Training
Training of single config could be run using the following commands:
```bash
# Training on Alesis3630 dataset
python train.py --config /home/yf/YF/ml/git/vca-comp-release/configs/num_bands_type_abilation/multiband_gru_1_band.yaml --dataset alesis3630

# Training on Alesis3630 dataset
python train.py --config /home/yf/YF/ml/git/vca-comp-release/configs/num_bands_type_abilation/multiband_gru_1_band.yaml --dataset nightshine
```

## Evaluation
Evaluation of all the presented models could be run using the following commands:

```bash
# Eval on Alesis3630 dataset
python test.py --config /home/yf/YF/ml/git/vca-comp-release/configs/num_bands_type_abilation/multiband_gru_1_band.yaml --dataset alesis3630

# Eval on Alesis3630 dataset
python test.py --config /home/yf/YF/ml/git/vca-comp-release/configs/num_bands_type_abilation/multiband_gru_1_band.yaml --dataset nightshine
```
The script applies the evaluation settings from the config and runs all selected models.

## Aknowledgemnts
This code is partially based on [Micro-TCN](https://github.com/csteinmetz1/micro-tcn) classes for data handling and modeling.

## Contributing
Pull requests are welcomed. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT