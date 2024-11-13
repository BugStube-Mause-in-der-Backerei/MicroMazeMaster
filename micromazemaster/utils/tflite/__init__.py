import torch
import torch.nn as nn
from micromazemaster.utils.logging import logger
from onnx2tf import convert


def convert_pytorch_to_onnx(model: nn.Module, input_tensor: torch.Tensor, output_onnx_file_path: str):

    # Set the model to evaluation mode
    model.eval()

    onnx_program = torch.onnx.dynamo_export(model, input_tensor)

    onnx_program.save(output_onnx_file_path)

    logger.info(f"Converted pytorch to onnx: {output_onnx_file_path}")


def convert_onnx_to_tflite(input_onnx_file_path: str, output_folder_path: str):

    convert(
        input_onnx_file_path=input_onnx_file_path,
        output_folder_path=output_folder_path,
        check_onnx_tf_outputs_elementwise_close_full=True,
    )

    logger.info(f"Converted onnx to tflite: {input_onnx_file_path} -> {output_folder_path}")


def convert_tflite_to_header(tflite_path: str, output_header_path: str):

    with open(tflite_path, "rb") as tflite_file:
        tflite_content = tflite_file.read()

    hex_lines = [
        ", ".join([f"0x{byte:02x}" for byte in tflite_content[i : i + 12]]) for i in range(0, len(tflite_content), 12)
    ]

    hex_array = ",\n  ".join(hex_lines)

    with open(output_header_path, "w") as header_file:

        header_file.write("alignas(16) const unsigned char g_model[] = {\n  ")
        header_file.write(f"{hex_array}\n")
        header_file.write("};\n\n")

    logger.info(f"Converted tflite to header: {tflite_path} -> {output_header_path}")
