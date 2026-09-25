# SOAR: High-Performance Resolution-Preserving Segmentation Framework

A dedicated, standalone C++20 / Vulkan Compute scientific semantic segmentation framework engineered specifically for native full-resolution imagery ($2048 \times 2048$) with extreme class imbalance and ultra-thin, curvilinear topological structures (e.g., solar filaments, retinal vessels, material cracks).

## Core Architectural Pillars

1. **Pure C++20 Standalone Runtime**: Zero PyTorch, zero LibTorch, zero CUDA vendor-lock. Native C++ implementation of the complete training and inference pipeline, autograd engine, loss functions, and dataset parsers.
2. **Cross-Vendor GPU Acceleration via Vulkan Compute**: Raw Vulkan compute API utilizing custom SPIR-V GLSL compute shaders. Runs across NVIDIA, AMD, Intel, and Apple Silicon (via MoltenVK), with automatic Google SwiftShader fallback for CPU execution.
3. **Deterministic Memory Sub-Allocation via `palloc`**: Custom low-level memory allocator managing Host/Staging arenas and sub-allocating Vulkan VRAM for strict $O(1)$ allocation resets with zero dynamic memory allocation jitter during training or inference.
4. **Strict Native Resolution ($2048 \times 2048$) Preservation**: Direct high-resolution feature preservation without lossy downsampling, built with GroupNorm for normalization stability on single-sample batches.
5. **Universal Dataset Support**: Native C++ COCO polygon JSON parser, YOLO segmentation text format parser, and scanline polygon rasterization.
6. **Unified Scientific Export**: Direct 1-indexed Run-Length Encoding (RLE) generation and 24-bit uncompressed BMP visualization export.
7. **Zero-Overhead Python Wrapper**: Lightweight Python ctypes bindings (`import soar`) and unified CLI (`cli.py` / `soar.bat` / `./soar.sh`) delegating to the native C++ engine.

---

## Project Structure

```
OPT-HQ-Net/
├── CMakeLists.txt              # Unified CMake build system (C++20)
├── palloc/                     # Custom low-level memory allocator submodule
├── configs/                    # Declarative YAML model & train configs
│   ├── default.yaml            # Master default training configuration
│   └── models/                 # Model scale specifications
│       ├── soar_nano1.yaml     # SOAR1-Nano
│       ├── soar_small1.yaml    # SOAR1-Small
│       ├── soar_medium1.yaml   # SOAR1-Medium
│       ├── soar_large1.yaml    # SOAR1-Large
│       └── soar_xlarge1.yaml   # SOAR1-XLarge
│
├── include/soar/               # C++ Header declarations
│   ├── api/c_api.h             # Pure C-ABI bindings for shared library (.dll / .so)
│   ├── core/                   # ConfigParser, Logger, Profiler
│   ├── data/                   # COCODataset, YOLODataset, ImageIO, PolygonRasterizer
│   ├── engine/                 # Trainer, Predictor, RLEEncoder
│   ├── losses/                 # BCEWithLogits, DiceLoss, CompositeLoss
│   ├── memory/palloc_allocator.hpp # palloc arena integration
│   ├── nn/                     # Layers (Conv2d, GroupNorm, SiLU, etc.), Blocks, SOARModel
│   ├── optim/                  # AdamW, CosineAnnealingLR
│   ├── shaders/                # Compiled SPIR-V bytecode headers (.spv.h)
│   ├── tensor/                 # Tensor, Shape, Dynamic Autograd DAG
│   └── vulkan/                 # DynamicLoader, VulkanContext, ComputePipeline, VulkanBuffer
│
├── shaders/                    # Raw GLSL compute shaders
│   ├── conv2d_1x1.comp         # Optimized 1x1 2D convolution
│   ├── conv2d_dw_3x3.comp      # 3x3 Depthwise separable convolution
│   ├── conv2d_dw_5x5.comp      # 5x5 Depthwise separable convolution
│   ├── conv2d_dw_7x7.comp      # 7x7 Depthwise separable convolution
│   ├── group_norm_stats.comp   # Parallel two-pass mean/variance calculation
│   ├── group_norm_affine_silu.comp # Fused affine transform with SiLU activation
│   ├── pixel_shuffle_2x.comp   # Sub-pixel convolution upsampling
│   ├── bilinear_resize.comp    # Resolution-preserving bilinear interpolation
│   ├── convex_fusion.comp      # Dynamic learned multi-scale feature weighting
│   ├── sigmoid.comp            # Numerically stable elementwise sigmoid
│   ├── tensor_add.comp         # Residual connection kernel
│   └── global_avg_pool.comp    # Squeeze-and-excitation reduction kernel
│
├── src/                        # C++ Implementation
│   ├── api/c_api.cpp           # Exported C interface implementation
│   ├── core/                   # Config parsing & core utilities
│   ├── data/                   # Dataset loaders & image decoding
│   ├── engine/                 # Training loop, evaluation, and inference
│   ├── losses/                 # Numerical loss gradients
│   ├── nn/                     # Forward/backward operators and models
│   ├── optim/                  # AdamW weight updates
│   ├── tensor/                 # Tensor memory and autograd engine
│   ├── vulkan/                 # Vulkan context, loader, and compute dispatch
│   └── main.cpp                # Native standalone CLI executable
│
├── tests/                      # Full C++ CTest unit and integration tests
│   ├── test_palloc_memory.cpp  # palloc sub-allocation & alignment tests
│   ├── test_vulkan_backend.cpp # Headless Vulkan compute & staging transfer tests
│   ├── test_operators.cpp      # Mathematical parity tests vs PyTorch equations
│   ├── test_full_engine.cpp    # Full model forward/backward/AdamW/serialization tests
│   └── test_data_pipeline.cpp  # COCO & YOLO dataset and polygon rasterizer tests
│
├── third_party/                # Vendored zero-install dependencies
│   ├── nlohmann/json.hpp       # Fast JSON parser for COCO annotations
│   ├── stb/stb_image.h         # Zero-dependency image decoding
│   └── vulkan/                 # Khronos Vulkan C headers
│
├── tools/                      # Build & maintenance scripts
│   └── compile_shaders.py      # GLSL to C SPIR-V header compiler
│
├── soar/                       # Thin Python ctypes wrapper package
│   ├── __init__.py             # Pythonic SOAR class wrapping libsoar_engine
│   └── cli.py                  # CLI dispatcher
│
├── cli.py                      # Root command-line interface
├── soar.bat                    # Windows native executable launcher
└── soar.sh                     # Linux / macOS native executable launcher
```

---

## Building from Source

### Prerequisites
- Modern C++20 compiler: GCC 11+, Clang 13+, or MSVC 2019+
- CMake 3.20+
- Vulkan SDK or runtime driver (NVIDIA / AMD / Intel driver, or MoltenVK, or SwiftShader)

### Build Steps

```bash
# 1. Clone repository with submodules
git clone --recursive https://github.com/pomagrenate/OPT-HQ-Net.git
cd OPT-HQ-Net

# 2. Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 3. Compile
cmake --build build --config Release -j
```

Upon compilation, the following artifacts are produced in `build/`:
- `soar_engine` (`soar_engine.exe`): Standalone native CLI application.
- `libsoar_engine` (`.dll` / `.so`): Shared C-ABI library for Python and external bindings.
- `libsoar_core.a`: Static C++ library.
- Unit test binaries (`test_palloc_memory`, `test_vulkan_backend`, `test_operators`, `test_full_engine`, `test_data_pipeline`).

### Running the Test Suite

```bash
ctest --test-dir build --output-on-failure
```

---

## Command-Line Usage

You can use the native executable directly (`build/soar_engine`), via wrapper scripts (`./soar.bat` on Windows or `./soar.sh` on Linux), or via `python cli.py`.

### 1. Training

Train natively in C++ on COCO or YOLO datasets:

```bash
# Using native CLI
./soar.bat train \
    --model configs/models/soar_nano1.yaml \
    --data data/dataset \
    --annotation-file data/dataset/annotations.json \
    --data-format coco \
    --epochs 50 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints

# Using Python CLI wrapper
python cli.py train \
    --model configs/models/soar_nano1.yaml \
    --data data/dataset \
    --annotation-file data/dataset/annotations.json \
    --format coco \
    --epochs 50 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints
```

### 2. High-Resolution Prediction & Mask Generation

Run inference on high-resolution images, generating prediction masks and RLE strings:

```bash
./soar.bat predict \
    --weights checkpoints/best.soar \
    --input test_images/sample.bmp \
    --output predictions/sample_mask.bmp \
    --threshold 0.5
```

### 3. Hardware Benchmark

Benchmark native $2048 \times 2048$ resolution-preserving throughput:

```bash
./soar.bat benchmark --variant nano --img-size 2048 2048
```

---

## Python API Usage

The Python package `soar` provides a direct, zero-overhead ctypes interface to the compiled native engine:

```python
import numpy as np
import soar

# 1. Initialize native C++ SOAR model
model = soar.SOAR(in_channels=1, num_classes=1, variant="nano")
print(f"Total model parameters: {model.parameter_count}")

# 2. Run C++ inference
image = np.ones((1, 2048, 2048), dtype=np.float32)
probs, binary_mask, rle_string = model.predict(image, threshold=0.5)

# 3. Run C++ autograd training step
mask = np.zeros((1, 2048, 2048), dtype=np.float32)
metrics = model.train_step(image, mask, lr=1e-3)
print(f"Loss: {metrics['loss']:.4f}, Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")

# 4. Save and load native binary weights (.soar)
model.save_weights("checkpoints/model.soar")
model.load_weights("checkpoints/model.soar")
```

---

## Mathematical Parity Guarantee

Every operator, activation, and loss has been mathematically verified against PyTorch reference equations:
- **Group Normalization**: Biased sample variance formulation matching `torch.nn.functional.group_norm` ($\text{Var}(x) = \frac{1}{N}\sum (x - \mu)^2$).
- **SiLU Activation**: $f(x) = x \cdot \sigma(x)$ with analytically verified gradient $f'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))$.
- **Upsampling**: Exact bilinear interpolation formula with `align_corners=False`.
- **Loss Functions**: Soft Dice loss with exact autograd gradient derivation matching continuous backpropagation dynamics.
