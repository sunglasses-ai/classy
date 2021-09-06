import numpy as np
import socket

from typing import Iterable


def flatten(lst: Iterable[list]) -> list:
    return [_e for sub_l in lst for _e in sub_l]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = np.random.uniform(-noise_value, noise_value)
    return max(1, value + noise)


def get_local_ip_address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    address = s.getsockname()[0]
    s.close()
    return address
