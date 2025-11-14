# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build configuration for Hand Gesture Music Controller.
Use alongside README build instructions to reproduce the distributable executable.
"""

from pathlib import Path
import shutil

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

project_root = Path.cwd()
assets_dir = project_root / "assets"
strip_available = shutil.which("strip") is not None  # Avoid strip warnings when binary is missing

app_datas = []
for asset_name in ("tray.png", "tray.ico"):
    asset_path = assets_dir / asset_name
    if asset_path.exists():
        app_datas.append((str(asset_path), "assets"))

icon_path = assets_dir / "app.ico"

EXCLUDED_DATA_DIR_NAMES = {"tests", "testdata", "__pycache__", "benchmark"}
EXCLUDED_HIDDENIMPORT_TOKENS = (".tests.", ".test.", "_test", ".benchmark", ".examples.")
EXCLUDED_MODULES = [
    "tensorflow",
    "tensorflow_intel",
    "tensorflow_io",
    "tensorflow_io_gcs",
    "torch",
    "torchaudio",
    "torchvision",
    "torchtext",
    "tflite_runtime",
    "keras",
    "onnx",
    "onnxruntime",
    "mediapipe.tasks.python.genai",
]

mediapipe_datas = [
    (src, dest)
    for src, dest in collect_data_files("mediapipe")
    if not any(part in EXCLUDED_DATA_DIR_NAMES for part in Path(src).parts)
]
mediapipe_binaries = collect_dynamic_libs("mediapipe")
mediapipe_hiddenimports = [
    name
    for name in collect_submodules("mediapipe")
    if not any(token in name for token in EXCLUDED_HIDDENIMPORT_TOKENS)
       and "mediapipe.tasks.python.genai" not in name
]

opencv_binaries = collect_dynamic_libs("cv2")

block_cipher = None

a = Analysis(
    ["hand_music_control.py"],
    pathex=[str(project_root)],
    binaries=mediapipe_binaries + opencv_binaries,
    datas=mediapipe_datas + app_datas,
    hiddenimports=mediapipe_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / "pyi_run_mediapipe_genai_fix.py")],
    excludes=EXCLUDED_MODULES + ["mediapipe.tasks.python.genai"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    exclude_binaries=True,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HandGestureMusicController",
    icon=str(icon_path) if icon_path.exists() else None,
    debug=False,
    bootloader_ignore_signals=False,
    strip=strip_available,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=strip_available,
    upx=True,
    upx_exclude=[],
    name="HandGestureMusicController",
)
