# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

BASE_DIR = os.path.abspath(SPECPATH)

datas = [
    (os.path.join(BASE_DIR, 'config'), 'config'),
    (os.path.join(BASE_DIR, 'assets'), 'assets'),
    (os.path.join(BASE_DIR, 'custom_filters'), 'custom_filters'),
    (os.path.join(BASE_DIR, 'saved_drawings'), 'saved_drawings'),
]
binaries = []
hiddenimports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    'mediapipe',
    'cv2',
    'numpy',
    'PIL',
    'scipy',
    'scipy.ndimage',
    'scipy.spatial',
]

# Collect mediapipe packages and models
mp_datas, mp_binaries, mp_hidden = collect_all('mediapipe')
datas += mp_datas
binaries += mp_binaries
hiddenimports += [h for h in mp_hidden if 'model_maker' not in h]

# Collect PyQt6 plugins and binaries
qt_datas, qt_binaries, qt_hidden = collect_all('PyQt6')
datas += qt_datas
binaries += qt_binaries
hiddenimports += qt_hidden

a = Analysis(
    ['main.py'],
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['mediapipe.model_maker', 'matplotlib', 'tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DoodleLens',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(BASE_DIR, 'assets', 'icons', 'app_icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='DoodleLens',
)
