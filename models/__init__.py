from .tcn.tcn import TCNModel
from .lstm_raw.lstm import LSTMModel
from .multiband.multiband_model import MultibandModelV2

models_map = {
    'tcn': TCNModel,
    'lstm': LSTMModel,
    'multiband': MultibandModelV2,
}