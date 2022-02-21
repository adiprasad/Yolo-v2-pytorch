from src.converter import TorchToTFLiteConverter
import json
import torch.nn as nn
from mcunet.tinynas.nn.networks.proxyless_nets import ProxylessNASNets
import torch


def run():
    with open("mcunet/assets/configs/mcunet-5fps_imagenet.json") as f:
        config = json.load(f)

    model = ProxylessNASNets.build_from_config(config)

    # model.anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
    #                 (11.2364, 10.0071)]

    #model.classifier = nn.Conv2d(160, len(model.anchors) * (5 + 5), 1, 1, 0, bias=False)

    torch_to_tf_converter = TorchToTFLiteConverter(save_conversion_log=False)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = model.to(device)

    torch_to_tf_converter.convert(model, "yolo_mcunet_converted_from_torch_5_classes.tflite")


if __name__ == "__main__":
    run()