# Redirect: config spostata al root del progetto.
# Questo file mantiene la compatibilità con gli import esistenti.
from configs.processes_config import *  # noqa: F401,F403
from configs.processes_config import _build_st_processes  # noqa: F401  (underscore excluded from *)
