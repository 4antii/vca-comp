from .vca_dataset import VCADataset

datasets_map = {
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
        'nightshine': {
            'dataset_class': VCADataset,
            'train_source': "path_to_soruces",
            'train_targets': "path_to_NightShine",
            'val_source': "path_to_soruces",
            'val_targets': "path_to_NightShine",
            'test_source': "path_to_soruces",
            'test_targets': "path_to_NightShine",
            'nparams': 4
        }}