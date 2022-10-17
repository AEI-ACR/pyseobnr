#!/usr/bin/env python
"""This file contains the default settings for various models
Each default setting is a *function* the resturns a dictionary.
To be used by models in models.py
"""

from typing import Any, Dict


def default_unfaitfulness_mode_by_mode_settings() -> Dict[Any, Any]:
    settings = dict(
        sigma=0.001,  # the sigma to use in likelihood
    )
    return settings
