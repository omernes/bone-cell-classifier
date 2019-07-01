from fastai.vision.models import DynamicUnet
from fastai.conv_learner import *
from fastai.dataset import *
from fastai.models.unet import *


# load defined model
def get_encoder(f, cut):
    base_model = (cut_model(f(True), cut))
    return nn.Sequential(*base_model)

f = resnet18
cut, cut_lr = model_meta[f]
encoder = get_encoder(f, cut)


unet_model = DynamicUnet(encoder=encoder, n_classes=6)
