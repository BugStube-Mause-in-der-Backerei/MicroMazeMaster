"""
MicroMazeMaster
"""

__version__ = "0.1.0"

cool_banner = rf"""

    __  ____                 __  ___                 __  ___           __           
   /  |/  (_)_____________  /  |/  /___ _____  ___  /  |/  /___ ______/ /____  _____
  / /|_/ / / ___/ ___/ __ \/ /|_/ / __ `/_  / / _ \/ /|_/ / __ `/ ___/ __/ _ \/ ___/
 / /  / / / /__/ /  / /_/ / /  / / /_/ / / /_/  __/ /  / / /_/ (__  ) /_/  __/ /    
/_/  /_/_/\___/_/   \____/_/  /_/\__,_/ /___/\___/_/  /_/\__,_/____/\__/\___/_/     

{__version__}
"""
from micromazemaster.utils.config import settings  # noqa
from rich import print  # noqa

print(cool_banner)
