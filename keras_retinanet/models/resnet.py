"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from tensorflow import keras
from tensorflow.keras.utils import get_file
from tensorflow.keras import applications as app

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class ResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ["resnet50", "resnet101", "resnet152"]
        backbone = self.backbone.split("_")[0]

        if backbone not in allowed_backbones:
            raise ValueError(
                "Backbone ('{}') not in allowed backbones ({}).".format(
                    backbone, allowed_backbones
                )
            )

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through
        the network.  """
        return preprocess_image(inputs, mode="caffe")


def _freeze_bn(model):
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, keras.layers.BatchNormalization):
            layer._per_input_updates = {}


def resnet_retinanet(
    num_classes, backbone="resnet50", inputs=None, modifier=None, **kwargs
):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None,
                None, 3)).
        modifier: A function handler which can modify the backbone before using
            it in retinanet (this can be used to freeze backbone layers for
            example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == "channels_first":
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == "resnet50":
        resnet = app.ResNet50(input_tensor=inputs, include_top=False)
        resnet_output_names = [
            "conv3_block4_out",
            "conv4_block6_out",
            "conv5_block3_out",
        ]
    elif backbone == "resnet101":
        resnet = app.ResNet101(input_tensor=inputs, include_top=False)
        resnet_output_names = [
            "conv3_block4_out",
            "conv4_block23_out",
            "conv5_block3_out",
        ]
    elif backbone == "resnet152":
        resnet = app.ResNet152(input_tensor=inputs, include_top=False)
        resnet_output_names = [
            "conv3_block8_out",
            "conv4_blocki36_out",
            "conv5_block3_out",
        ]
    else:
        raise ValueError("Backbone ('{}') is invalid.".format(backbone))

    _freeze_bn(resnet)

    if False:
        print("Backbone summary:")
        resnet.summary()
        from tensorflow.keras.utils import plot_model
        plot_model(resnet, to_file="model.png")

    backbone_outputs = [
        resnet.get_layer(layer_name).output for layer_name in resnet_output_names
    ]

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    # create the full model
    return retinanet.retinanet(
        inputs=inputs,
        num_classes=num_classes,
        backbone_layers=backbone_outputs,
        **kwargs
    )


def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(
        num_classes=num_classes, backbone="resnet50", inputs=inputs, **kwargs
    )


def resnet101_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(
        num_classes=num_classes, backbone="resnet101", inputs=inputs, **kwargs
    )


def resnet152_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(
        num_classes=num_classes, backbone="resnet152", inputs=inputs, **kwargs
    )
