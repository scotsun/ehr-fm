# MIMIC-IV Data Curation Package

from mimic4.definitions import defs

# Export defs at module level for dagster CLI
__all__ = ["defs"]

# Make defs directly accessible as mimic4.defs
import sys
sys.modules[__name__].defs = defs

