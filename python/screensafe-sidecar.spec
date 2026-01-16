# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for ScreenSafe Python sidecar

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect EasyOCR data files and models
easyocr_datas = collect_data_files('easyocr')
easyocr_hiddenimports = collect_submodules('easyocr')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=easyocr_datas,
    hiddenimports=[
        'cv2',
        'numpy',
        'torch',
        'torchvision',
        'PIL',
        'websockets',
        'asyncio',
        *easyocr_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'notebook',
        'jupyter',
    ],
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
    name='screensafe-sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
