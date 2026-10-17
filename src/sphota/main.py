import pydeepity as pd
import numpy as np
import torch

from e46_data import get_omniglot_stream
from prototype_memory import FrozenPrototypeMemory

def main() -> None:
    print("Hello world")

    stream = get_omniglot_stream()

    pcn = dy.FullConvPCN()

if __name__ == "__main__":
    main()