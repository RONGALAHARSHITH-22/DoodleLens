# ✨ DoodleLens — AI-Powered Air Drawing & Face Filter Studio

A real-time computer vision application that lets you draw in the air using hand gestures and apply AR-style face filters using your webcam — no touch, no mouse, no paid APIs required.

---

## 🚀 Features

| Feature | Status |
|---------|--------|
| Air Drawing with finger tracking | ✅ |
| 10 creative brush types (neon, rainbow, fire, electric…) | ✅ |
| Gesture controls (draw, pause, clear, snapshot, mode switch) | ✅ |
| Real-time face tracking via MediaPipe | ✅ |
| 10 built-in AR face filters (angel, demon, galaxy, crown…) | ✅ |
| Convert air drawings → custom face filters | ✅ |
| Save and reload custom filters | ✅ |
| Undo / Redo / Clear / Save drawing | ✅ |
| Dark neon premium UI | ✅ |
| Shape recognition (circle, rectangle, triangle, line…) | ✅ |
| All local, no cloud, no paid APIs | ✅ |

---

## 📋 Requirements

- Windows 10 or Windows 11
- Python 3.9+ (tested on 3.9.13)
- A webcam (built-in or USB)

---

## ⚡ Quick Setup (Windows)

### Step 1: Clone / Download the project

```batch
cd "C:\Games\Custome Effects filter\AirCanvasFX"
```

### Step 2: (Optional but recommended) Create a virtual environment

```batch
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```batch
pip install -r requirements.txt
```

> **Note:** This installs opencv-python, mediapipe, PyQt6, numpy, Pillow, and scipy.

### Step 4: Launch the application

```batch
python main.py
```

---

## 🌐 Running on Web & Vercel

DoodleLens also includes an in-browser Web Studio powered by MediaPipe JS:

### Run Locally:
```batch
cd web
python serve.py
```
This serves the application locally at `http://localhost:8000` and automatically opens your browser.

### Deploy to Vercel:
The repository includes a pre-configured `vercel.json`. Connect this GitHub repository on [vercel.com](https://vercel.com) and click **Deploy**.

---

## 📦 Building Standalone Windows Executable (.exe)

To bundle DoodleLens into a standalone desktop application that runs without installing Python or dependencies:

```batch
pip install pyinstaller
python -m PyInstaller DoodleLens.spec --noconfirm
```
The output executable will be created in `dist/DoodleLens/DoodleLens.exe`.

---

## 🎮 How to Use

### Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ Index finger up | Draw continuously |
| ✋ Open palm (all fingers extended) | Pause drawing (lift pen) |
| ✊ Closed fist | Clear canvas |
| 👍 Thumbs up | Take a screenshot |
| ✌️ Two fingers (index + middle) | Switch between Drawing ↔ Filter mode |
| 🤌 Pinch (thumb + index close) | Select / Activate tool |

> **Tip:** Hold each gesture steady for ~8 frames to activate it. This prevents accidental triggering.

### Air Drawing

1. Click **▶ Start Camera**
2. Raise your **index finger** in front of the webcam
3. Move your finger to draw — lines appear in real time
4. Use **Open Palm** to pause drawing and reposition your hand
5. Pick brush types, colors, and sizes in the **left toolbar**

### Face Filters

1. Click **▶ Start Camera**
2. Click **🎭 Filter Mode** in the bottom bar OR use the ✌️ gesture
3. Select any filter from the **right panel** gallery
4. Filters automatically track your face, eyes, and head movement
5. Adjust **Intensity** with the slider

### Converting a Drawing to a Face Filter

1. Draw something in **Drawing Mode** (e.g., a crown, glasses, horns, halo)
2. In the right panel, select an **anchor** (e.g., "Forehead", "Above Head", "Both Eyes")
3. Click **🎨 Convert & Save Filter**
4. Enter a name for your filter
5. Select it from the **Custom Filters** dropdown
6. Click **▶ Apply Custom** to wear your own drawing as a face filter!

---

## 🖌️ Brush Types

| Brush | Description |
|-------|-------------|
| **Neon Pen** | Glowing neon line with additive blending |
| **Rainbow Brush** | Smoothly cycling color spectrum |
| **Laser Trail** | Bright white core with colored glow |
| **Glowing Marker** | Wide soft-glow strokes |
| **Particle Brush** | Exploding particle trails |
| **Sparkle Trail** | Star-shaped sparkle particles |
| **Fire Trail** | Warm orange/red ember particles |
| **Electric Lightning** | Jittery electric spark strokes |
| **Soft Pastel** | Transparent watercolor-style strokes |
| **Rainbow Spiral** | Rotating hue-shifted colors |

---

## 🎭 Built-in Face Filters

| Filter | Description |
|--------|-------------|
| **Neon Aura** | Pulsing neon rings and particles around face |
| **Cyberpunk Mask** | Futuristic lines, visor effect, corner brackets |
| **Angel Mode** | Golden halo above head with floating sparkles |
| **Demon Mode** | Red glowing horns + ember particles |
| **Star Eyes** | Rotating stars over each eye |
| **Heart Eyes** | Pulsing animated hearts on eyes |
| **Galaxy Face** | Cosmic particle rings around face |
| **Rainbow Crown** | Multi-color crown following head |
| **Sketch Face** | Pencil-line contour overlay |
| **Custom Drawing** | Your own air-drawn creation |

---

## 🧪 Running Tests

```batch
python -m pytest tests/test_core.py -v
```

Or directly:

```batch
python tests/test_core.py
```

### Manual Tests

For features that require a webcam:

1. **Hand tracking**: Start camera, move hand in front — verify green dot follows fingertip
2. **Drawing**: Raise index finger, draw a large circle — verify it's smooth and continuous
3. **Tracking loss**: Quickly hide hand from camera — verify no line connects when hand reappears
4. **Face filters**: Apply "Angel Mode" and tilt your head — verify halo follows head angle
5. **Custom filter**: Draw a moustache shape → Convert → Apply as filter on nose anchor

---

## 📁 Project Structure

```
DoodleLens/
├── main.py                     # Entry point
├── requirements.txt
├── README.md
├── config/
│   └── settings.json           # App settings
├── src/
│   ├── app.py                  # Central controller
│   ├── camera_manager.py       # Webcam capture thread
│   ├── hand_tracker.py         # MediaPipe Hands
│   ├── gesture_recognition.py  # Gesture classifier + debouncer
│   ├── drawing_engine.py       # Canvas + 10 brush types
│   ├── shape_recognition.py    # Contour shape detection
│   ├── face_tracker.py         # MediaPipe Face Mesh
│   ├── face_filter_engine.py   # 10 AR face filters
│   ├── custom_filter_manager.py# Save/load custom filters
│   └── ui/
│       └── main_window.py      # PyQt6 premium UI
├── assets/
│   ├── filters/                # Filter assets
│   └── icons/
├── saved_drawings/             # Saved PNGs
├── custom_filters/             # User-created filter PNGs + index.json
└── tests/
    └── test_core.py
```

---

## 🔧 Troubleshooting

### Camera not detected
- Ensure no other application is using the webcam (Teams, Zoom, etc.)
- Try changing the camera index in `config/settings.json` (`"device_index": 1`)
- Run the app as Administrator if camera access is blocked

### MediaPipe import error
```batch
pip uninstall mediapipe -y
pip install mediapipe
```

### PyQt6 issue on Windows
```batch
pip install --upgrade PyQt6 PyQt6-Qt6 PyQt6-sip
```

### Low FPS performance
- In `config/settings.json`, set `"quality": "low"` and reduce `"max_particles": 50`
- Reduce camera resolution: `"width": 640, "height": 480`
- Ensure good, consistent lighting for better tracking

### Hand not detected
- Ensure good lighting facing your hand
- Keep hand within webcam frame
- Adjust `hand_detection_confidence` lower (e.g. `0.5`) in settings.json

### Face filter not following correctly
- Face needs to be well-lit and clearly visible
- Try `face_detection_confidence: 0.4` if detection is poor
- Face mesh tracking works best when facing camera head-on

---

## 📝 Known Limitations

1. **3D face tracking**: Uses 2D landmark projection; no full 3D head pose. Filters maintain 2D stability but don't rotate in 3D perspective.
2. **Iris landmarks disabled**: For maximum compatibility, iris-specific landmarks are not used. Eye center is interpolated from surrounding landmarks.
3. **Multi-hand**: Currently tracks one hand. Two-hand support can be enabled in settings (`"max_hands": 2`).
4. **FPS**: Performance depends on hardware. Lower-end laptops may run at 15–20 FPS instead of 30.

---

## 📄 License

MIT License — free for personal and educational use.

---

*Built with ❤️ using Python, OpenCV, MediaPipe, and PyQt6*
