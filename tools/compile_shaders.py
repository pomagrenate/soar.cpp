#!/usr/bin/env python3
"""
Compile GLSL compute shaders to SPIR-V C header byte arrays.
"""
import subprocess
import sys
from pathlib import Path

GLSLC_PATHS = [
    r"E:\Admin\AppData\Local\Android\sdk\ndk\27.1.12297006\shader-tools\windows-x86_64\glslc.exe",
    "glslc",
]

def find_glslc():
    for p in GLSLC_PATHS:
        try:
            res = subprocess.run([p, "--version"], capture_output=True, text=True)
            if res.returncode == 0:
                return p
        except Exception:
            continue
    raise RuntimeError("glslc compiler not found")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    shaders_dir = repo_root / "shaders"
    out_dir = repo_root / "include" / "soar" / "shaders"
    out_dir.mkdir(parents=True, exist_ok=True)

    glslc = find_glslc()
    print(f"Using glslc: {glslc}")

    shader_files = sorted(list(shaders_dir.glob("*.comp")))
    if not shader_files:
        print("No .comp shader files found in", shaders_dir)
        return 1

    headers = []
    for sf in shader_files:
        stem = sf.stem
        spv_h = out_dir / f"{stem}.spv.h"
        cmd = [glslc, str(sf), "-mfmt=c", "-o", str(spv_h)]
        print(f"Compiling {sf.name} -> {spv_h.name}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error compiling {sf.name}:\n{res.stderr}")
            return 1
        headers.append((stem, spv_h.name))

    # Generate master header
    master_path = out_dir / "spv_shaders.hpp"
    with open(master_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n\n")
        f.write("#include <cstdint>\n")
        f.write("#include <span>\n\n")
        f.write("namespace soar::shaders {\n\n")

        for stem, fname in headers:
            var_name = f"SPV_{stem.upper()}"
            f.write(f"inline constexpr uint32_t {var_name}[] = \n")
            f.write(f'#include "{fname}"\n')
            f.write(";\n\n")
            f.write(f"inline std::span<const uint32_t> get_{stem}() {{\n")
            f.write(f"    return {var_name};\n")
            f.write("}\n\n")

        f.write("} // namespace soar::shaders\n")

    print(f"Successfully generated master shader header: {master_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
