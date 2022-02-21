from datetime import datetime
import io

import onnx

import os
from onnx_tf.backend import prepare

import shutil

import tempfile
import torch
import tensorflow as tf
from torchvision import datasets

from src.log_utils import stdout_redirector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''

To set this up :-


conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
conda install onnx
pip install onnxruntime
pip install onnx-tf
pip install --upgrade Pillow 

'''


class TFLiteConverter:

    def __init__(self, converter):
        self.converter = converter
        self.converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

    @classmethod
    def from_saved_model(cls, saved_model_dir):
        return TFLiteConverter(tf.lite.TFLiteConverter.from_saved_model(saved_model_dir))

    @classmethod
    def from_keras_model(cls, keras_model):
        return TFLiteConverter(tf.lite.TFLiteConverter.from_keras_model(keras_model))

    @classmethod
    def from_concrete_functions(cls, funcs):
        return TFLiteConverter(tf.lite.TFLiteConverter.from_concrete_functions([funcs]))

    def add_weight_optimization(self):
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def add_variable_optimization(self, repr_dataset_path, data_transforms=None):
        dataset = datasets.ImageFolder(repr_dataset_path, transform=data_transforms)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, drop_last=False,
                                                  num_workers=4)
        list_of_batch_tensors = []

        for i, (inputs, labels) in enumerate(data_loader):
            list_of_batch_tensors.append(inputs)

        cat_tensors = torch.cat(list_of_batch_tensors, dim=0)
        np_tensor = cat_tensors.numpy()
        tf_tensor = tf.convert_to_tensor(np_tensor)

        def representative_data_gen():
            for input_value in tf.data.Dataset.from_tensor_slices(tf_tensor).batch(1).take(100):
                # Model has only one input so each data point has one element.
                yield [input_value]

        self.converter.representative_dataset = representative_data_gen

    def add_input_output_optimization(self, repr_dataset_path=None, data_transforms=None):
        if not self.converter.representative_dataset:
            self.add_variable_optimization(repr_dataset_path, data_transforms)

        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        self.converter.inference_input_type = tf.uint8
        self.converter.inference_output_type = tf.uint8

    def convert(self):
        return self.converter.convert()


class TorchToTFLiteConverter:
    TMP_DIR = tempfile.gettempdir()

    INTERMEDIATE_ONNX_MODEL = os.path.join(TMP_DIR, "intermediate_onnx_model.onnx")
    INTERMEDIATE_TF_CKPT = os.path.join(TMP_DIR, "intermediate_tf_ckpt.pb")

    def __init__(self, input_dims=(3, 96, 96), onnx_do_constant_folding=True,
                 repr_dataset_path=None, repr_dataset_transforms=None, save_conversion_log=False, log_file_dir=None):
        assert len(input_dims) == 3
        assert type(onnx_do_constant_folding) == bool
        # assert save_conversion_log and log_file_dir, "save_conversion_log should be set AND log_file_dir " \
        #                                              "should be specified to save log!"

        self.input_dims = input_dims  # Model input in CHW (channel first)
        self.constant_folding = onnx_do_constant_folding
        self.repr_dataset_path = repr_dataset_path
        self.repr_dataset_transforms = repr_dataset_transforms
        self.save_conversion_log = save_conversion_log
        self.log_file_dir = log_file_dir

        if repr_dataset_transforms and not repr_dataset_path:
            raise RuntimeError("repr_dataset_path is required along with repr_dataset_transforms!")

    def convert(self, torch_model, tflite_output_path):
        assert type(tflite_output_path) == str, "tflite_output_path should be a string!"
        assert isinstance(torch_model, torch.nn.Module), "torch_model should be an instance of torch.nn.Module"
        f = io.BytesIO()

        if self.save_conversion_log:
            with stdout_redirector(f):
                self.__convert_steps(torch_model, tflite_output_path)

            self.__log_output_to_file(f.getvalue().decode('utf-8'), tflite_output_path)
        else:
            self.__convert_steps(torch_model, tflite_output_path)

    def __convert_steps(self, torch_model, tflite_output_path):
        self._convert_to_onnx(torch_model)
        self._convert_from_onnx_to_tflite(tflite_output_path)

    def __log_output_to_file(self, output, tflite_output_path):
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

        if not os.path.exists(self.log_file_dir):
            os.makedirs(self.log_file_dir)

        log_file = os.path.join(self.log_file_dir, "{}_{}".format(tflite_output_path, date_time))
        converter_variable_dict = vars(self)

        with open(log_file, "w") as f:
            f.write(
                "================================== Converter details ==================================" + os.linesep)
            for key in converter_variable_dict.keys():
                f.write("{} : {}".format(key, converter_variable_dict[key]) + os.linesep)
            f.write("================================== TF output log ==================================" + os.linesep)
            f.write(output)

    def _convert_to_onnx(self, torch_model):
        if os.path.exists(self.INTERMEDIATE_ONNX_MODEL):
            os.remove(self.INTERMEDIATE_ONNX_MODEL)

        torch_model.eval()
        batch_size = 1

        torch_input = torch.randn((batch_size,) + self.input_dims, requires_grad=True)

        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        torch_input = torch_input.to(device)

        # Export the model
        torch.onnx.export(torch_model,  # model being run
                          torch_input,  # model input (or a tuple for multiple inputs)
                          self.INTERMEDIATE_ONNX_MODEL,
                          # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=self.constant_folding,
                          # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})

        onnx_model = onnx.load(self.INTERMEDIATE_ONNX_MODEL)
        onnx.checker.check_model(onnx_model)

    def _convert_from_onnx_to_tflite(self, tflite_output_path):
        onnx_model = onnx.load(self.INTERMEDIATE_ONNX_MODEL)  # load onnx model

        # prepare function converts an ONNX model to an internel representation
        # of the computational graph called TensorflowRep and returns
        # the converted representation.
        tf_rep = prepare(onnx_model)  # creating TensorflowRep object

        # export_graph function obtains the graph proto corresponding to the ONNX
        # model associated with the backend representation and serializes
        # to a protobuf file.
        if os.path.exists(self.INTERMEDIATE_TF_CKPT):
            shutil.rmtree(self.INTERMEDIATE_TF_CKPT)
        tf_rep.export_graph(self.INTERMEDIATE_TF_CKPT)

        # Convert the model
        converter = TFLiteConverter.from_saved_model(self.INTERMEDIATE_TF_CKPT)  # path to the SavedModel directory
        converter.add_weight_optimization()

        if self.repr_dataset_path:
            converter.add_variable_optimization(repr_dataset_path=self.repr_dataset_path,
                                                data_transforms=self.repr_dataset_transforms)
            converter.add_input_output_optimization()

        tflite_model = converter.convert()

        # Save the model.
        with open(tflite_output_path, 'wb') as f:
            f.write(tflite_model)


# if __name__ == "__main__":
#     '''
#     ofa_mbv3 = OFAMobileNetV3(dropout_rate=0, width_mult=1.0, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6],
#                               depth_list=[2, 3, 4])
#
#     checkpoint = torch.load('.torch/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0', map_location=torch.device('cpu'))
#     # print(checkpoint.keys())
#     ofa_mbv3.load_state_dict(checkpoint['state_dict'])
#
#
#     _ = ofa_mbv3.sample_active_subnet()
#     model = ofa_mbv3.get_active_subnet()
#
#     batch_size = 1
#     # set the model to inference mode
#     model.eval()
#
#     # weight_quantization(model, 8, 20)
#
#     # Input to the model
#     x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
#     torch_out = model(x)
#
#     # Export the model
#     torch.onnx.export(model,  # model being run
#                       x,  # model input (or a tuple for multiple inputs)
#                       "intermediate_onnx_model.onnx",
#                       # where to save the model (can be a file or file-like object)
#                       export_params=True,  # store the trained parameter weights inside the model file
#                       opset_version=8,  # the ONNX version to export the model to
#                       do_constant_folding=True,  # whether to execute constant folding for optimization
#                       input_names=['input'],  # the model's input names
#                       output_names=['output'],  # the model's output names
#                       dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#                                     'output': {0: 'batch_size'}})
#
#     onnx_model = onnx.load("intermediate_onnx_model.onnx")
#     onnx.checker.check_model(onnx_model)
#
#     ort_session = onnxruntime.InferenceSession("intermediate_onnx_model.onnx")
#
#     def to_numpy(tensor):
#         return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
#     # compute ONNX Runtime output prediction
#     ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
#     ort_outs = ort_session.run(None, ort_inputs)
#
#     # compare ONNX Runtime and PyTorch results
#     np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
#
#     print("Exported model has been tested with ONNXRuntime, and the result looks good!")
#
#     '''
#     ofa_mbv3 = OFAMobileNetV3(dropout_rate=0, width_mult=1.0, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6],
#                               depth_list=[2, 3, 4])
#
#     model_ckpt_path = ".torch/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0"
#
#     checkpoint = torch.load(model_ckpt_path, map_location=torch.device('cpu'))
#
#     ofa_mbv3.load_state_dict(checkpoint['state_dict'])
#
#     converted_successfully = 0
#     not_converted = 0
#
#     subnet_config = ofa_mbv3.sample_active_subnet()
#     subnet = ofa_mbv3.get_active_subnet()
#
#     converter = TorchToTFLiteConverter(save_conversion_log=True, log_file_dir="log_dir_dummy")
#     converter.convert(subnet, "subnet_from_mbv3_ofa.tflite")
#
#     '''
#     for i in range(10):
#         try:
#             subnet_config = ofa_mbv3.sample_active_subnet()
#             subnet = ofa_mbv3.get_active_subnet()
#
#             converter = TorchToTFLiteConverter(subnet, "subnet_from_mbv3_ofa.tflite", save_conversion_log=False)
#
#             converter.convert()
#
#             converted_successfully += 1
#         except Exception as e:
#             not_converted += 1
#
#
#     print("Converted : {}/10".format(converted_successfully))
#     print("Not Converted : {}/10".format(not_converted))
#     '''