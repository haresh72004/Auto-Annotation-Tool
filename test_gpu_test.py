import sys


def main():
    print("=" * 60)
    print("AUTO ANNOTATION TOOL - GPU TEST")
    print("=" * 60)

    print(f"Python: {sys.version}")

    try:
        import torch
    except ImportError:
        print("\nERROR: PyTorch is not installed.")
        print("Install/configure PyTorch before continuing.")
        return

    print(f"PyTorch: {torch.__version__}")

    cuda_available = torch.cuda.is_available()

    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("\nWARNING:")
        print("CUDA is NOT available.")
        print("Grounding DINO will not use your RTX GPU.")
        print("\nCheck your PyTorch CUDA installation.")
        return

    print(f"CUDA version: {torch.version.cuda}")

    gpu_count = torch.cuda.device_count()

    print(f"GPU count: {gpu_count}")

    for index in range(gpu_count):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

    print("\nGPU TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
