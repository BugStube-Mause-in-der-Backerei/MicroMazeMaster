import typer

from micromazemaster.tools.cli.convert import convert_cli
from micromazemaster.tools.cli.train import train_cli
from micromazemaster.utils.config import settings

micromazemaster_cli = typer.Typer(
    invoke_without_command=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=settings.typer.enable_pretty_exception,
    pretty_exceptions_show_locals=settings.typer.show_pretty_exception_local,
)
micromazemaster_cli.add_typer(train_cli, name="train")
micromazemaster_cli.add_typer(convert_cli, name="convert")


@micromazemaster_cli.callback()
def micrmoazemaster_callback(ctx: typer.Context):
    pass


if __name__ == "__main__":
    micromazemaster_cli()
