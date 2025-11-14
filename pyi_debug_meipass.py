"""PyInstaller runtime hook to print _MEIPASS extraction contents for debugging.

When included with --runtime-hook, this prints the temp extraction path and
the top-level files/directories so we can confirm whether MediaPipe model
files were bundled and extracted for onefile builds.
"""
import sys
import os

def _dump_meipass():
    meipass = getattr(sys, "_MEIPASS", None)
    try:
        print("[pyi_debug_meipass] _MEIPASS:", meipass)
        if meipass and os.path.exists(meipass):
            for root, dirs, files in os.walk(meipass):
                # Only print the first few directories/files to avoid spamming
                rel = os.path.relpath(root, meipass)
                print(f"[pyi_debug_meipass] {rel}")
                for f in files[:20]:
                    print("   ", f)
                # limit depth
                if rel.count(os.sep) > 2:
                    continue
    except Exception as e:
        print("[pyi_debug_meipass] error:", e)

_dump_meipass()
