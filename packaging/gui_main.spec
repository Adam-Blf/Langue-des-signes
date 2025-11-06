# PyInstaller specification for building the standalone GUI executable.
# Generated so contributors can run: pyinstaller packaging/gui_main.spec

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

SPEC_PATH = Path(globals().get("__file__", sys.argv[0])).resolve()
PROJECT_ROOT = SPEC_PATH.parent.parent
MAIN_SCRIPT = PROJECT_ROOT / "gui_main.py"
DATA_FILES = []

MODEL_PATH = PROJECT_ROOT / "machine_learning" / "model.pkl"
if MODEL_PATH.exists():
    DATA_FILES.append((str(MODEL_PATH), "machine_learning"))

MEDIAPIPE_DATAS = collect_data_files("mediapipe")
MEDIAPIPE_HIDDEN = collect_submodules("mediapipe")
SKLEARN_DATAS = collect_data_files("sklearn")
SKLEARN_HIDDEN = collect_submodules("sklearn")

a = Analysis(
    [str(MAIN_SCRIPT)],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=DATA_FILES + MEDIAPIPE_DATAS + SKLEARN_DATAS,
    hiddenimports=MEDIAPIPE_HIDDEN + SKLEARN_HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="lsf-detector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
