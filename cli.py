#!/usr/bin/env python3
"""
SOAR CLI forwarder delegating to the native C++20 soar_engine binary.
"""
import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    exe = root / "build" / ("soar_engine.exe" if sys.platform == "win32" else "soar_engine")
    if not exe.exists():
        exe = root.parent / "build" / ("soar_engine.exe" if sys.platform == "win32" else "soar_engine")

    if not exe.exists():
        print(f"[ERROR] Native executable not found at {exe}. Please build the project first.", file=sys.stderr)
        sys.exit(1)

    cmd = [str(exe)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
