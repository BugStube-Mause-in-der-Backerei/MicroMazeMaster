import shutil
from pathlib import PurePath

import torch
import typer
from typing_extensions import Annotated

from micromazemaster.utils.config import settings
from micromazemaster.utils.logging import logger
from micromazemaster.utils.preprocessing import create_tmp_dir
from micromazemaster.utils.tflite import (
    convert_onnx_to_tflite,
    convert_pytorch_to_onnx,
    convert_tflite_to_header,
)

convert_cli = typer.Typer(no_args_is_help=True)


@convert_cli.callback()
def convert_callback():
    """Micromazemaster module for converting model for use on microcontroller"""
    pass


@convert_cli.command("tflite")
def cmd_convert_tflite(
    model_type: Annotated[str, typer.Argument(help="The type of model to convert. (evolution / dql)")],
    model_path: Annotated[str, typer.Argument(help="The path to the model to convert.")],
):
    """Convert a model to tflite format"""

    WORKING_DIR = create_tmp_dir()

    model_name = PurePath(model_path).stem

    if model_type == "evolution":
        from micromazemaster.utils.evolution import MazeSolver

        model = MazeSolver(
            settings.evolution.input_size, settings.evolution.hidden_size, settings.evolution.output_size
        )

        model.load_state_dict(torch.load(WORKING_DIR.parent.joinpath(model_path)))

        torch_input = torch.randn(1, settings.evolution.input_size)

        logger.info("Model loaded successfully.")

    elif model_type == "dql":
        return
    else:
        logger.error("Invalid model type")
        raise ValueError("Invalid model type")

    convert_pytorch_to_onnx(model, torch_input, str(WORKING_DIR.joinpath(f"{model_name}.onnx")))

    convert_onnx_to_tflite(str(WORKING_DIR.joinpath(f"{model_name}.onnx")), str(WORKING_DIR.joinpath(f"{model_name}")))

    convert_tflite_to_header(
        str(WORKING_DIR.joinpath(f"{model_name}").joinpath(f"{model_name}_float32.tflite")),
        str(WORKING_DIR.joinpath(f"{model_name}_float32.h")),
    )

    shutil.copyfile(
        str(WORKING_DIR.joinpath(f"{model_name}_float32.h")),
        str(WORKING_DIR.parent.joinpath(f"{model_name}_float32.h")),
    )

    shutil.rmtree(WORKING_DIR)
    logger.info("Model converted successfully.")
    logger.info(f"Header file saved at {WORKING_DIR.parent.joinpath(f'{model_name}.h')}")
