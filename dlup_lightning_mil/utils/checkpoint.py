# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# As taken from VISSL

from typing import Dict, Any


def replace_module_prefix(
    state_dict: Dict[str, Any], prefix: str, replace_with: str = "", ignore_prefix: str = ""
):
    """
    Remove prefixes in a state_dict needed when loading models that are not VISSL
    trained models.

    Specify the prefix in the keys that should be removed.

    Added by DLM contributors: ignore_prefix is used to ignore certain keys in the state dict
    """
    state_dict = {
        (key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items() if ((not key.startswith(ignore_prefix)) or ignore_prefix == "")
    }
    return state_dict
