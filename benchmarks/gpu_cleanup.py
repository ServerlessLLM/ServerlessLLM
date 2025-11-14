#!/usr/bin/env python3
"""GPU memory cleanup utility for benchmark runs."""

import torch
import gc


def main():
    print("GPU memory allocated before cleanup: %.2f GB" % (torch.cuda.memory_allocated()/1024**3))
    print("GPU memory reserved before cleanup: %.2f GB" % (torch.cuda.memory_reserved()/1024**3))

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    print("GPU memory allocated after cleanup: %.2f GB" % (torch.cuda.memory_allocated()/1024**3))
    print("GPU memory reserved after cleanup: %.2f GB" % (torch.cuda.memory_reserved()/1024**3))


if __name__ == "__main__":
    main()
