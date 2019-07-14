from os import getenv

from keras.engine.saving import load_model

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_loss_function.keras_ssd_loss import SSDLoss

MODEL_PATH = getenv("MODEL_PATH")
# MODEL_PATH = "ssd300_pascal_07+12_epoch-80_loss-4.4898_val_loss-5.6198.h5"

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model = load_model(MODEL_PATH, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'compute_loss': ssd_loss.compute_loss})

weights = model.save_weights(f"{MODEL_PATH[:-3]}_weights-only.h5")


