import math
from typing import Dict

import torch


def firstblock_derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["block_activated_steps"][-1] - current["block_activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["firstblock_max_order"]):
        if (cache_dic["cache"]["firstblock_hidden"].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - cache_dic["cache"]["firstblock_hidden"][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"]["firstblock_hidden"] = updated_taylor_factors


def firstblock_taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion error.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current["step"] - current["block_activated_steps"][-1]
    output = 0
    # if len(cache_dic['cache']['firstblock_hidden']) == 1:
    #     return output
    for i in range(len(cache_dic["cache"]["firstblock_hidden"])):
        
        output += (1 / math.factorial(i)) * cache_dic["cache"]["firstblock_hidden"][i] * (x**i)
       
    return output


def step_derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (cache_dic["cache"]["hidden"].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
          
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - cache_dic["cache"]["hidden"][i]
            ) / difference_distance
        else:
            break
    
    
    cache_dic["cache"]["hidden"] = updated_taylor_factors


def step_taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion error.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current["step"] - current["activated_steps"][-1]

    output = 0
    # if len(cache_dic['cache']['hidden']) == 1:
    #     return output
    for i in range(len(cache_dic["cache"]["hidden"])):

        output += (1 / math.factorial(i)) * cache_dic["cache"]["hidden"][i] * (x**i)

    return output


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (
            cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]].get(i, None) is not None
        ) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i]
                - cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion error.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current["step"] - current["activated_steps"][-1]

    output = 0

    for i in range(len(cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]])):
        output += (
            (1 / math.factorial(i))
            * cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            * (x**i)
        )

    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current["step"] == 0) and (cache_dic["taylor_cache"]):
        cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = {}

        