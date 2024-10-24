import sys

from dynaconf import Dynaconf
from dynaconf.vendor.box import BoxKeyError
from rich import print

from micromazemaster.utils.logging import logger

try:
    settings = Dynaconf(
        envvar_prefix="MICROMAZEMASTER",
        env_switcher="MICROMAZEMASTER_ENV",
        settings_files=["settings.toml", ".secrets.toml"],
        environments=True,
        default_env="default",
        merge_enabled=True,
    )

except BoxKeyError or AttributeError:
    logger.error("❌ Invalid config!!")
    sys.exit(1)

logger.setLevel(settings.logging.level)

if settings.logging.level == "DEBUG":
    if settings.root_path_for_dynaconf:
        logger.debug("Config root path: %s", settings.root_path_for_dynaconf)
    print("CONFIG".center(50, "="))
    print(settings.as_dict())
else:
    logger.info("[ Using Environment ]\t %s", settings.current_env)
