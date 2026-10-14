/**
 * ✨ DoodleLens Web Studio - AI Air Drawing & AR Face Filters
 * Runs in-browser using MediaPipe Hands, FaceMesh, and HTML5 Canvas.
 */

// ── State Management ────────────────────────────────────────────────────────
const state = {
  cameraActive: false,
  cameraPaused: false,
  mirror: true,
  showLandmarks: true,
  autoSnapShapes: false,
  
  mode: 'drawing', // 'drawing' | 'filter'
  activeBrush: 'neon',
  brushColor: '#00f0ff',
  brushSize: 12,
  glowIntensity: 80,
  eraserActive: false,
  
  activeFilter: 'angel',
  filterIntensity: 1.0,
  
  // Custom Filter Studio
  customFilters: [],
  activeCustomFilter: null,
  
  // Drawing Canvas Data
  strokes: [],          // Array of { type, color, size, glow, points: [{x, y, t}] }
  redoStack: [],
  currentStroke: null,
  
  // Gesture Debouncing
  rawGesture: 'NONE',
  stableGesture: 'NONE',
  gestureCount: 0,
  GESTURE_DEBOUNCE: 6,
  
  // Performance
  fps: 0,
  frameCount: 0,
  lastFpsUpdate: performance.now(),
};

// ── DOM Elements ────────────────────────────────────────────────────────────
const video = document.getElementById('webcam');
const stageCanvas = document.getElementById('stage-canvas');
const ctx = stageCanvas.getContext('2d');

const camStatusPill = document.getElementById('cam-status-pill');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const fpsCounter = document.getElementById('fps-counter');
const gesturePill = document.getElementById('current-gesture');
const badgeIcon = document.getElementById('badge-icon');
const badgeName = document.getElementById('badge-name');
const hudModeTag = document.getElementById('hud-mode-tag');
const camPlaceholder = document.getElementById('cam-placeholder');
const flashOverlay = document.getElementById('flash-overlay');

// Controls
const btnToggleCam = document.getElementById('btn-toggle-cam');
const btnPauseCam = document.getElementById('btn-pause-cam');
const btnToggleMode = document.getElementById('btn-toggle-mode');
const btnStartHero = document.getElementById('btn-start-camera-hero');
const btnSnapshot = document.getElementById('btn-snapshot');
const btnDownloadArt = document.getElementById('btn-download-art');
const btnUndo = document.getElementById('btn-undo');
const btnRedo = document.getElementById('btn-redo');
const btnClear = document.getElementById('btn-clear');
const btnEraser = document.getElementById('btn-eraser');

const brushSizeSlider = document.getElementById('brush-size');
const sizeValLabel = document.getElementById('size-val');
const sizePreviewDot = document.getElementById('size-preview-dot');
const glowSlider = document.getElementById('glow-intensity');
const glowValLabel = document.getElementById('glow-val');
const customColorInput = document.getElementById('custom-color-input');

const filterIntensitySlider = document.getElementById('filter-intensity');
const filterIntensityVal = document.getElementById('filter-intensity-val');

const chkLandmarks = document.getElementById('chk-landmarks');
const chkShapes = document.getElementById('chk-shapes');
const chkMirror = document.getElementById('chk-mirror');

// Modal
const btnGuide = document.getElementById('btn-guide');
const modalGuide = document.getElementById('modal-guide');
const btnCloseModal = document.getElementById('btn-close-modal');
const btnGotIt = document.getElementById('btn-got-it');

// Custom Filter Studio
const btnConvertFilter = document.getElementById('btn-convert-filter');
const customFilterNameInput = document.getElementById('custom-filter-name');
const anchorSelect = document.getElementById('anchor-select');
const customFilterList = document.getElementById('custom-filter-list');

// ── MediaPipe Instances ─────────────────────────────────────────────────────
let mpHands = null;
let mpFaceMesh = null;
let latestHandLandmarks = null;
let latestFaceLandmarks = null;

// Initialize MediaPipe models
function initMediaPipe() {
  // Hands
  mpHands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });
  mpHands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
  mpHands.onResults((results) => {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      latestHandLandmarks = results.multiHandLandmarks[0];
    } else {
      latestHandLandmarks = null;
    }
  });

  // Face Mesh
  mpFaceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });
  mpFaceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  mpFaceMesh.onResults((results) => {
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      latestFaceLandmarks = results.multiFaceLandmarks[0];
    } else {
      latestFaceLandmarks = null;
    }
  });
}

// ── Camera Lifecycle ────────────────────────────────────────────────────────
async function startCamera() {
  try {
    statusText.textContent = "Starting Camera...";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: false
    });
    
    video.srcObject = stream;
    await video.play();
    
    state.cameraActive = true;
    state.cameraPaused = false;
    
    stageCanvas.width = video.videoWidth || 1280;
    stageCanvas.height = video.videoHeight || 720;
    
    statusDot.className = "status-dot active";
    statusText.textContent = "LIVE";
    btnToggleCam.textContent = "⏹ Stop Camera";
    btnToggleCam.classList.remove('btn-primary');
    btnToggleCam.classList.add('btn-danger');
    camPlaceholder.classList.add('hidden');
    
    // Start main render loop
    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error("Camera access error:", err);
    alert(`Could not start webcam: ${err.message}. Please allow camera permissions.`);
    statusText.textContent = "Camera Blocked";
  }
}

function stopCamera() {
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null;
  }
  state.cameraActive = false;
  statusDot.className = "status-dot";
  statusText.textContent = "Camera Off";
  btnToggleCam.textContent = "▶ Start Camera";
  btnToggleCam.classList.remove('btn-danger');
  btnToggleCam.classList.add('btn-primary');
  camPlaceholder.classList.remove('hidden');
  latestHandLandmarks = null;
  latestFaceLandmarks = null;
}

function togglePause() {
  if (!state.cameraActive) return;
  state.cameraPaused = !state.cameraPaused;
  if (state.cameraPaused) {
    statusDot.className = "status-dot paused";
    statusText.textContent = "PAUSED";
    btnPauseCam.textContent = "▶ Resume";
  } else {
    statusDot.className = "status-dot active";
    statusText.textContent = "LIVE";
    btnPauseCam.textContent = "⏸ Pause";
  }
}

// ── Main Processing & Render Loop ───────────────────────────────────────────
let processingFrame = false;

async function renderLoop() {
  if (!state.cameraActive) return;

  const now = performance.now();
  state.frameCount++;
  if (now - state.lastFpsUpdate >= 1000) {
    state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFpsUpdate));
    fpsCounter.textContent = state.fps;
    state.frameCount = 0;
    state.lastFpsUpdate = now;
  }

  if (!state.cameraPaused && video.readyState >= 2) {
    // 1. Process MediaPipe in parallel
    if (!processingFrame) {
      processingFrame = true;
      Promise.all([
        mpHands ? mpHands.send({ image: video }) : Promise.resolve(),
        mpFaceMesh ? mpFaceMesh.send({ image: video }) : Promise.resolve()
      ]).finally(() => {
        processingFrame = false;
      });
    }

    // 2. Classify gesture
    if (latestHandLandmarks) {
      const g = classifyGesture(latestHandLandmarks);
      updateGestureState(g);
    } else {
      updateGestureState('NONE');
    }

    // 3. Update drawing strokes from fingertip
    handleAirDrawing();
  }

  // 4. Render all visual layers to Canvas
  drawScene();

  requestAnimationFrame(renderLoop);
}

// ── Gesture Classification & Debouncing ─────────────────────────────────────
function classifyGesture(landmarks) {
  // Landmarks: 4=Thumb tip, 8=Index tip, 12=Middle tip, 16=Ring tip, 20=Pinky tip
  // Base joints: 2, 6, 10, 14, 18
  const isUp = (tipIdx, pipIdx) => landmarks[tipIdx].y < landmarks[pipIdx].y;
  
  const indexUp = isUp(8, 6);
  const middleUp = isUp(12, 10);
  const ringUp = isUp(16, 14);
  const pinkyUp = isUp(20, 18);
  
  // Thumb up check (distance from wrist and angle)
  const thumbUp = landmarks[4].y < landmarks[3].y && landmarks[4].y < landmarks[2].y;

  // 1. Thumbs Up
  if (thumbUp && !indexUp && !middleUp && !ringUp && !pinkyUp) {
    return 'THUMBS_UP';
  }

  // 2. Closed Fist
  if (!indexUp && !middleUp && !ringUp && !pinkyUp && !thumbUp) {
    return 'CLOSED_FIST';
  }

  // 3. Two Fingers (Peace)
  if (indexUp && middleUp && !ringUp && !pinkyUp) {
    return 'PEACE';
  }

  // 4. Open Palm
  if (indexUp && middleUp && ringUp && pinkyUp) {
    return 'OPEN_PALM';
  }

  // 5. Point & Draw (Index only)
  if (indexUp && !middleUp && !ringUp) {
    return 'POINT_DRAW';
  }

  return 'NONE';
}

function updateGestureState(detected) {
  if (detected === state.rawGesture) {
    state.gestureCount++;
    if (state.gestureCount >= state.GESTURE_DEBOUNCE && state.stableGesture !== detected) {
      state.stableGesture = detected;
      onGestureTriggered(detected);
    }
  } else {
    state.rawGesture = detected;
    state.gestureCount = 1;
  }

  // Update UI Pills
  const names = {
    'POINT_DRAW': '☝️ Drawing',
    'OPEN_PALM': '✋ Paused',
    'CLOSED_FIST': '✊ Clear Canvas',
    'THUMBS_UP': '👍 Screenshot',
    'PEACE': '✌️ Mode Switch',
    'NONE': 'None'
  };
  gesturePill.textContent = names[state.stableGesture] || 'None';
  
  if (state.stableGesture !== 'NONE') {
    badgeIcon.textContent = names[state.stableGesture].split(' ')[0];
    badgeName.textContent = names[state.stableGesture].split(' ').slice(1).join(' ');
  } else {
    badgeIcon.textContent = '👋';
    badgeName.textContent = 'Ready';
  }
}

let lastSnapshotTime = 0;
let lastModeSwitchTime = 0;

function onGestureTriggered(gesture) {
  const now = performance.now();

  if (gesture === 'CLOSED_FIST') {
    clearCanvas();
  } else if (gesture === 'THUMBS_UP') {
    if (now - lastSnapshotTime > 2000) {
      takeSnapshot();
      lastSnapshotTime = now;
    }
  } else if (gesture === 'PEACE') {
    if (now - lastModeSwitchTime > 1500) {
      toggleAppMode();
      lastModeSwitchTime = now;
    }
  }
}

// ── Air Drawing Handling ────────────────────────────────────────────────────
function handleAirDrawing() {
  if (!latestHandLandmarks) {
    endStroke();
    return;
  }

  const tip = latestHandLandmarks[8]; // Index tip
  const w = stageCanvas.width;
  const h = stageCanvas.height;
  
  // Account for mirror
  const x = state.mirror ? (1 - tip.x) * w : tip.x * w;
  const y = tip.y * h;

  if (state.stableGesture === 'POINT_DRAW') {
    if (!state.currentStroke) {
      state.currentStroke = {
        type: state.eraserActive ? 'eraser' : state.activeBrush,
        color: state.brushColor,
        size: state.brushSize,
        glow: state.glowIntensity,
        points: []
      };
      state.strokes.push(state.currentStroke);
      state.redoStack = [];
    }
    state.currentStroke.points.push({ x, y, t: performance.now() });
  } else {
    endStroke();
  }
}

function endStroke() {
  if (state.currentStroke) {
    if (state.currentStroke.points.length < 2) {
      state.strokes.pop();
    }
    state.currentStroke = null;
  }
}

function clearCanvas() {
  state.strokes = [];
  state.redoStack = [];
  state.currentStroke = null;
}

function undoStroke() {
  if (state.strokes.length > 0) {
    state.redoStack.push(state.strokes.pop());
  }
}

function redoStroke() {
  if (state.redoStack.length > 0) {
    state.strokes.push(state.redoStack.pop());
  }
}

// ── Scene Rendering (Video + Filters + Drawing) ─────────────────────────────
function drawScene() {
  const w = stageCanvas.width;
  const h = stageCanvas.height;

  ctx.clearRect(0, 0, w, h);

  // 1. Render Video Feed
  if (video.readyState >= 2) {
    ctx.save();
    if (state.mirror) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();
  }

  // 2. Render AR Face Filter
  if (state.mode === 'filter' && latestFaceLandmarks) {
    renderFaceFilter(ctx, latestFaceLandmarks, w, h);
  }

  // 3. Render Custom Filter if active
  if (state.mode === 'filter' && state.activeCustomFilter && latestFaceLandmarks) {
    renderCustomFilter(ctx, state.activeCustomFilter, latestFaceLandmarks, w, h);
  }

  // 4. Render Air Drawing Canvas Strokes
  renderStrokes(ctx);

  // 5. Render Tracking Landmarks (if enabled)
  if (state.showLandmarks) {
    renderLandmarksOverlay(ctx, w, h);
  }
}

// ── Brush Renderers ─────────────────────────────────────────────────────────
function renderStrokes(targetCtx) {
  for (const stroke of state.strokes) {
    if (!stroke.points || stroke.points.length < 2) continue;

    targetCtx.save();
    targetCtx.lineCap = 'round';
    targetCtx.lineJoin = 'round';

    if (stroke.type === 'eraser') {
      targetCtx.globalCompositeOperation = 'destination-out';
      targetCtx.lineWidth = stroke.size * 2;
      targetCtx.beginPath();
      targetCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
      for (let i = 1; i < stroke.points.length; i++) {
        targetCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
      }
      targetCtx.stroke();
    } else if (stroke.type === 'neon') {
      // Glow pass
      targetCtx.shadowColor = stroke.color;
      targetCtx.shadowBlur = (stroke.glow / 100) * 32;
      targetCtx.strokeStyle = stroke.color;
      targetCtx.lineWidth = stroke.size;
      drawSmoothPath(targetCtx, stroke.points);

      // Bright white core
      targetCtx.shadowBlur = 0;
      targetCtx.strokeStyle = '#ffffff';
      targetCtx.lineWidth = Math.max(2, stroke.size * 0.35);
      drawSmoothPath(targetCtx, stroke.points);
    } else if (stroke.type === 'rainbow') {
      for (let i = 1; i < stroke.points.length; i++) {
        const hue = (i * 12 + performance.now() * 0.1) % 360;
        targetCtx.strokeStyle = `hsl(${hue}, 100%, 60%)`;
        targetCtx.shadowColor = `hsl(${hue}, 100%, 50%)`;
        targetCtx.shadowBlur = (stroke.glow / 100) * 20;
        targetCtx.lineWidth = stroke.size;
        targetCtx.beginPath();
        targetCtx.moveTo(stroke.points[i - 1].x, stroke.points[i - 1].y);
        targetCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
        targetCtx.stroke();
      }
    } else if (stroke.type === 'fire') {
      for (let i = 1; i < stroke.points.length; i++) {
        const p = stroke.points[i];
        const rad = stroke.size * (0.8 + Math.sin(i) * 0.4);
        const grad = targetCtx.createRadialGradient(p.x, p.y, 1, p.x, p.y, rad * 1.5);
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(0.3, '#f59e0b');
        grad.addColorStop(0.8, '#ef4444');
        grad.addColorStop(1, 'rgba(239, 68, 68, 0)');
        targetCtx.fillStyle = grad;
        targetCtx.beginPath();
        targetCtx.arc(p.x, p.y - Math.sin(i) * 4, rad, 0, Math.PI * 2);
        targetCtx.fill();
      }
    } else if (stroke.type === 'electric') {
      targetCtx.shadowColor = '#38bdf8';
      targetCtx.shadowBlur = 18;
      targetCtx.strokeStyle = '#38bdf8';
      targetCtx.lineWidth = Math.max(2, stroke.size * 0.6);
      targetCtx.beginPath();
      targetCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
      for (let i = 1; i < stroke.points.length; i++) {
        const jitterX = (Math.random() - 0.5) * 8;
        const jitterY = (Math.random() - 0.5) * 8;
        targetCtx.lineTo(stroke.points[i].x + jitterX, stroke.points[i].y + jitterY);
      }
      targetCtx.stroke();
    } else if (stroke.type === 'sparkle') {
      targetCtx.strokeStyle = stroke.color;
      targetCtx.lineWidth = stroke.size * 0.5;
      drawSmoothPath(targetCtx, stroke.points);
      // Sparkle stars
      targetCtx.fillStyle = '#ffffff';
      for (let i = 0; i < stroke.points.length; i += 4) {
        drawStar(targetCtx, stroke.points[i].x, stroke.points[i].y, 4, stroke.size * 1.2, stroke.size * 0.4);
      }
    } else if (stroke.type === 'calligraphy') {
      targetCtx.fillStyle = stroke.color;
      for (let i = 1; i < stroke.points.length; i++) {
        const p1 = stroke.points[i - 1];
        const p2 = stroke.points[i];
        const angle = Math.PI / 4;
        const offX = Math.cos(angle) * (stroke.size * 0.8);
        const offY = Math.sin(angle) * (stroke.size * 0.8);
        targetCtx.beginPath();
        targetCtx.moveTo(p1.x - offX, p1.y - offY);
        targetCtx.lineTo(p1.x + offX, p1.y + offY);
        targetCtx.lineTo(p2.x + offX, p2.y + offY);
        targetCtx.lineTo(p2.x - offX, p2.y - offY);
        targetCtx.closePath();
        targetCtx.fill();
      }
    } else if (stroke.type === 'dots') {
      targetCtx.fillStyle = stroke.color;
      targetCtx.shadowColor = stroke.color;
      targetCtx.shadowBlur = 12;
      for (let i = 0; i < stroke.points.length; i += 2) {
        targetCtx.beginPath();
        targetCtx.arc(stroke.points[i].x, stroke.points[i].y, stroke.size * 0.6, 0, Math.PI * 2);
        targetCtx.fill();
      }
    } else {
      // Classic Solid
      targetCtx.strokeStyle = stroke.color;
      targetCtx.lineWidth = stroke.size;
      drawSmoothPath(targetCtx, stroke.points);
    }

    targetCtx.restore();
  }
}

function drawSmoothPath(c, pts) {
  if (pts.length < 2) return;
  c.beginPath();
  c.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i].x + pts[i + 1].x) / 2;
    const my = (pts[i].y + pts[i + 1].y) / 2;
    c.quadraticCurveTo(pts[i].x, pts[i].y, mx, my);
  }
  c.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  c.stroke();
}

function drawStar(c, cx, cy, spikes, outerRadius, innerRadius) {
  let rot = Math.PI / 2 * 3;
  let x = cx;
  let y = cy;
  let step = Math.PI / spikes;

  c.beginPath();
  c.moveTo(cx, cy - outerRadius);
  for (let i = 0; i < spikes; i++) {
    x = cx + Math.cos(rot) * outerRadius;
    y = cy + Math.sin(rot) * outerRadius;
    c.lineTo(x, y);
    rot += step;

    x = cx + Math.cos(rot) * innerRadius;
    y = cy + Math.sin(rot) * innerRadius;
    c.lineTo(x, y);
    rot += step;
  }
  c.lineTo(cx, cy - outerRadius);
  c.closePath();
  c.fill();
}

// ── Built-in AR Face Filters ────────────────────────────────────────────────
function renderFaceFilter(c, lm, w, h) {
  const getPt = (idx) => {
    const pt = lm[idx];
    const x = state.mirror ? (1 - pt.x) * w : pt.x * w;
    const y = pt.y * h;
    return { x, y };
  };

  const forehead = getPt(10);
  const noseTip = getPt(1);
  const leftEye = getPt(33);
  const rightEye = getPt(263);
  const leftCheek = getPt(234);
  const rightCheek = getPt(454);

  // Compute face orientation angle and scale
  const eyeDx = rightEye.x - leftEye.x;
  const eyeDy = rightEye.y - leftEye.y;
  const angle = Math.atan2(eyeDy, eyeDx);
  const faceWidth = Math.hypot(eyeDx, eyeDy);
  const scale = (faceWidth / 120) * state.filterIntensity;

  c.save();

  if (state.activeFilter === 'angel') {
    // 😇 Angel Halo
    c.translate(forehead.x, forehead.y - 80 * scale);
    c.rotate(angle);
    c.scale(scale, scale * 0.35);

    c.shadowColor = '#fef08a';
    c.shadowBlur = 30;
    c.strokeStyle = '#fef08a';
    c.lineWidth = 14;
    c.beginPath();
    c.arc(0, 0, 80, 0, Math.PI * 2);
    c.stroke();

    c.strokeStyle = '#ffffff';
    c.lineWidth = 6;
    c.stroke();
  } else if (state.activeFilter === 'demon') {
    // 😈 Cyber Devil Horns
    c.shadowColor = '#ef4444';
    c.shadowBlur = 24;
    c.fillStyle = '#ef4444';

    // Left Horn
    c.save();
    c.translate(leftCheek.x - 20 * scale, forehead.y - 10 * scale);
    c.rotate(angle - 0.4);
    c.scale(scale, scale);
    c.beginPath();
    c.moveTo(0, 0);
    c.quadraticCurveTo(-30, -50, -10, -90);
    c.quadraticCurveTo(10, -50, 20, 0);
    c.closePath();
    c.fill();
    c.restore();

    // Right Horn
    c.save();
    c.translate(rightCheek.x + 20 * scale, forehead.y - 10 * scale);
    c.rotate(angle + 0.4);
    c.scale(scale, scale);
    c.beginPath();
    c.moveTo(0, 0);
    c.quadraticCurveTo(30, -50, 10, -90);
    c.quadraticCurveTo(-10, -50, -20, 0);
    c.closePath();
    c.fill();
    c.restore();
  } else if (state.activeFilter === 'crown') {
    // 👑 Royal Crown
    c.translate(forehead.x, forehead.y - 45 * scale);
    c.rotate(angle);
    c.scale(scale, scale);

    c.shadowColor = '#fbbf24';
    c.shadowBlur = 25;
    c.fillStyle = 'rgba(251, 191, 36, 0.85)';
    c.strokeStyle = '#fef08a';
    c.lineWidth = 4;

    c.beginPath();
    c.moveTo(-70, 0);
    c.lineTo(-90, -60);
    c.lineTo(-45, -25);
    c.lineTo(0, -85);
    c.lineTo(45, -25);
    c.lineTo(90, -60);
    c.lineTo(70, 0);
    c.closePath();
    c.fill();
    c.stroke();

    // Crown jewels
    c.fillStyle = '#ec4899';
    c.beginPath();
    c.arc(0, -85, 8, 0, Math.PI * 2);
    c.arc(-90, -60, 6, 0, Math.PI * 2);
    c.arc(90, -60, 6, 0, Math.PI * 2);
    c.fill();
  } else if (state.activeFilter === 'visor') {
    // 🕶️ Cyber HUD Visor
    const eyeMidX = (leftEye.x + rightEye.x) / 2;
    const eyeMidY = (leftEye.y + rightEye.y) / 2;
    c.translate(eyeMidX, eyeMidY);
    c.rotate(angle);
    c.scale(scale, scale);

    c.shadowColor = '#06b6d4';
    c.shadowBlur = 24;
    c.fillStyle = 'rgba(6, 182, 212, 0.28)';
    c.strokeStyle = '#06b6d4';
    c.lineWidth = 3;

    c.beginPath();
    c.roundRect(-100, -22, 200, 44, 12);
    c.fill();
    c.stroke();

    // HUD Scanline
    const scanOffset = (performance.now() * 0.08) % 40 - 20;
    c.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    c.lineWidth = 2;
    c.beginPath();
    c.moveTo(-90, scanOffset);
    c.lineTo(90, scanOffset);
    c.stroke();
  } else if (state.activeFilter === 'catears') {
    // 🐱 Pink Cat Ears
    c.shadowColor = '#ec4899';
    c.shadowBlur = 20;
    c.fillStyle = 'rgba(236, 72, 153, 0.85)';
    c.strokeStyle = '#ffffff';
    c.lineWidth = 3;

    // Left Ear
    c.save();
    c.translate(leftCheek.x, forehead.y - 30 * scale);
    c.rotate(angle - 0.2);
    c.scale(scale, scale);
    c.beginPath();
    c.moveTo(-35, 10);
    c.lineTo(-25, -60);
    c.lineTo(15, -10);
    c.closePath();
    c.fill();
    c.stroke();
    c.restore();

    // Right Ear
    c.save();
    c.translate(rightCheek.x, forehead.y - 30 * scale);
    c.rotate(angle + 0.2);
    c.scale(scale, scale);
    c.beginPath();
    c.moveTo(35, 10);
    c.lineTo(25, -60);
    c.lineTo(-15, -10);
    c.closePath();
    c.fill();
    c.stroke();
    c.restore();
  } else if (state.activeFilter === 'galaxy') {
    // 🌌 Nebula Aura
    const t = performance.now() * 0.002;
    c.translate(noseTip.x, noseTip.y);
    c.rotate(angle);
    c.scale(scale, scale);

    for (let i = 0; i < 18; i++) {
      const dist = 90 + Math.sin(t + i) * 35;
      const a = (i / 18) * Math.PI * 2 + t * 0.5;
      const px = Math.cos(a) * dist;
      const py = Math.sin(a) * dist * 1.3;
      c.fillStyle = i % 2 === 0 ? '#a855f7' : '#06b6d4';
      c.shadowColor = c.fillStyle;
      c.shadowBlur = 16;
      c.beginPath();
      c.arc(px, py, 6 + Math.sin(t + i) * 3, 0, Math.PI * 2);
      c.fill();
    }
  } else if (state.activeFilter === 'lightning') {
    // ⚡ Lightning Arc
    c.shadowColor = '#38bdf8';
    c.shadowBlur = 20;
    c.strokeStyle = '#e0f2fe';
    c.lineWidth = 3 * scale;
    c.beginPath();
    c.moveTo(leftCheek.x, forehead.y);
    c.lineTo((leftCheek.x + forehead.x) / 2 + (Math.random() - 0.5) * 20, forehead.y - 30 * scale);
    c.lineTo(forehead.x, forehead.y - 15 * scale);
    c.lineTo((rightCheek.x + forehead.x) / 2 + (Math.random() - 0.5) * 20, forehead.y - 30 * scale);
    c.lineTo(rightCheek.x, forehead.y);
    c.stroke();
  } else if (state.activeFilter === 'mask') {
    // 🎭 Neon Tribal Mask
    c.translate(noseTip.x, noseTip.y);
    c.rotate(angle);
    c.scale(scale, scale);
    c.shadowColor = '#a855f7';
    c.shadowBlur = 18;
    c.strokeStyle = '#c084fc';
    c.lineWidth = 3;

    // Cheek spirals
    c.beginPath();
    c.arc(-45, 10, 20, 0, Math.PI * 1.5);
    c.stroke();
    c.beginPath();
    c.arc(45, 10, 20, Math.PI * 0.5, Math.PI * 2);
    c.stroke();
  }

  c.restore();
}

// ── Custom Filter Conversion & Rendering ────────────────────────────────────
function convertDrawingToFilter() {
  if (state.strokes.length === 0) {
    alert("Please draw something in Drawing Mode first (e.g. glasses, horns, or a crown) before converting!");
    return;
  }

  // Calculate bounding box of all strokes
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const s of state.strokes) {
    for (const p of s.points) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
  }

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const bw = Math.max(20, maxX - minX);
  const bh = Math.max(20, maxY - minY);

  // Normalize points relative to center (-0.5 to 0.5)
  const normalizedStrokes = state.strokes.map(s => ({
    type: s.type,
    color: s.color,
    size: s.size,
    glow: s.glow,
    points: s.points.map(p => ({
      x: (p.x - cx) / bw,
      y: (p.y - cy) / bh
    }))
  }));

  const filterName = customFilterNameInput.value.trim() || `Custom Filter #${state.customFilters.length + 1}`;
  const anchor = anchorSelect.value;

  const newFilter = {
    id: Date.now(),
    name: filterName,
    anchor: anchor,
    width: bw,
    height: bh,
    strokes: normalizedStrokes
  };

  state.customFilters.push(newFilter);
  state.activeCustomFilter = newFilter;
  saveCustomFiltersToStorage();
  renderCustomFilterList();

  // Clear name input & switch to filter mode
  customFilterNameInput.value = "";
  setAppMode('filter');
  alert(`🎉 "${filterName}" created and active! It will now track your face.`);
}

function renderCustomFilter(c, filter, lm, w, h) {
  const getPt = (idx) => {
    const pt = lm[idx];
    const x = state.mirror ? (1 - pt.x) * w : pt.x * w;
    const y = pt.y * h;
    return { x, y };
  };

  // Select anchor point
  let anchorPt = getPt(1); // Default nose
  if (filter.anchor === 'forehead') anchorPt = getPt(10);
  else if (filter.anchor === 'eyes') anchorPt = { x: (getPt(33).x + getPt(263).x) / 2, y: (getPt(33).y + getPt(263).y) / 2 };
  else if (filter.anchor === 'mouth') anchorPt = getPt(13);
  else if (filter.anchor === 'above_head') {
    const fh = getPt(10);
    anchorPt = { x: fh.x, y: fh.y - 60 };
  }

  const leftEye = getPt(33);
  const rightEye = getPt(263);
  const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
  const faceWidth = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
  const scale = (faceWidth / 120) * state.filterIntensity;

  c.save();
  c.translate(anchorPt.x, anchorPt.y);
  c.rotate(angle);
  c.scale(scale, scale);

  for (const s of filter.strokes) {
    c.save();
    c.strokeStyle = s.color;
    c.lineWidth = s.size;
    c.shadowColor = s.color;
    c.shadowBlur = (s.glow / 100) * 20;
    c.lineCap = 'round';
    c.lineJoin = 'round';

    c.beginPath();
    const p0 = { x: s.points[0].x * filter.width, y: s.points[0].y * filter.height };
    c.moveTo(p0.x, p0.y);
    for (let i = 1; i < s.points.length; i++) {
      const px = s.points[i].x * filter.width;
      const py = s.points[i].y * filter.height;
      c.lineTo(px, py);
    }
    c.stroke();
    c.restore();
  }

  c.restore();
}

function saveCustomFiltersToStorage() {
  try {
    localStorage.setItem('doodlelens_custom_filters', JSON.stringify(state.customFilters));
  } catch (e) {
    console.warn("Storage save error:", e);
  }
}

function loadCustomFiltersFromStorage() {
  try {
    const raw = localStorage.getItem('doodlelens_custom_filters');
    if (raw) {
      state.customFilters = JSON.parse(raw);
      renderCustomFilterList();
    }
  } catch (e) {}
}

function renderCustomFilterList() {
  if (state.customFilters.length === 0) {
    customFilterList.innerHTML = `<div class="empty-list-msg">No custom filters yet. Draw and convert one!</div>`;
    return;
  }

  customFilterList.innerHTML = state.customFilters.map(f => `
    <div class="custom-filter-item">
      <span>🎨 <b>${escapeHtml(f.name)}</b> (${f.anchor})</span>
      <div style="display:flex; gap:6px;">
        <button class="btn btn-primary" style="padding:4px 8px; font-size:11px;" onclick="applyCustomFilterById(${f.id})">
          ${state.activeCustomFilter?.id === f.id ? 'Active' : 'Apply'}
        </button>
        <button class="btn btn-danger" style="padding:4px 8px; font-size:11px;" onclick="deleteCustomFilterById(${f.id})">
          🗑️
        </button>
      </div>
    </div>
  `).join('');
}

window.applyCustomFilterById = function(id) {
  state.activeCustomFilter = state.customFilters.find(f => f.id === id) || null;
  setAppMode('filter');
  renderCustomFilterList();
};

window.deleteCustomFilterById = function(id) {
  state.customFilters = state.customFilters.filter(f => f.id !== id);
  if (state.activeCustomFilter?.id === id) state.activeCustomFilter = null;
  saveCustomFiltersToStorage();
  renderCustomFilterList();
};

function escapeHtml(str) {
  return str.replace(/[&<>'"]/g, tag => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[tag] || tag));
}

// ── Landmarks Overlay ───────────────────────────────────────────────────────
function renderLandmarksOverlay(c, w, h) {
  // Hand landmarks
  if (latestHandLandmarks) {
    c.save();
    c.fillStyle = '#06b6d4';
    c.shadowColor = '#06b6d4';
    c.shadowBlur = 8;
    for (const pt of latestHandLandmarks) {
      const x = state.mirror ? (1 - pt.x) * w : pt.x * w;
      const y = pt.y * h;
      c.beginPath();
      c.arc(x, y, 3, 0, Math.PI * 2);
      c.fill();
    }
    // Fingertip glow
    const tip = latestHandLandmarks[8];
    const tx = state.mirror ? (1 - tip.x) * w : tip.x * w;
    const ty = tip.y * h;
    c.fillStyle = '#ec4899';
    c.shadowColor = '#ec4899';
    c.shadowBlur = 16;
    c.beginPath();
    c.arc(tx, ty, 7, 0, Math.PI * 2);
    c.fill();
    c.restore();
  }

  // Face mesh highlights
  if (latestFaceLandmarks && state.mode === 'filter') {
    c.save();
    c.fillStyle = 'rgba(155, 92, 246, 0.4)';
    const sampleIndices = [10, 1, 33, 263, 61, 291];
    for (const idx of sampleIndices) {
      const pt = latestFaceLandmarks[idx];
      const x = state.mirror ? (1 - pt.x) * w : pt.x * w;
      const y = pt.y * h;
      c.beginPath();
      c.arc(x, y, 3.5, 0, Math.PI * 2);
      c.fill();
    }
    c.restore();
  }
}

// ── Snapshot & Save Artwork ─────────────────────────────────────────────────
function takeSnapshot() {
  flashOverlay.classList.add('active');
  setTimeout(() => flashOverlay.classList.remove('active'), 250);

  const link = document.createElement('a');
  link.download = `DoodleLens_Snapshot_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  link.href = stageCanvas.toDataURL('image/png');
  link.click();
}

function downloadArtworkOnly() {
  const offscreen = document.createElement('canvas');
  offscreen.width = stageCanvas.width;
  offscreen.height = stageCanvas.height;
  const oCtx = offscreen.getContext('2d');
  
  // Render dark gradient background
  const grad = oCtx.createLinearGradient(0, 0, offscreen.width, offscreen.height);
  grad.addColorStop(0, '#0a0a14');
  grad.addColorStop(1, '#121026');
  oCtx.fillStyle = grad;
  oCtx.fillRect(0, 0, offscreen.width, offscreen.height);
  
  // Render strokes
  renderStrokes(oCtx);

  const link = document.createElement('a');
  link.download = `DoodleLens_Artwork_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  link.href = offscreen.toDataURL('image/png');
  link.click();
}

// ── App Mode & Tabs ─────────────────────────────────────────────────────────
function setAppMode(newMode) {
  state.mode = newMode;
  if (newMode === 'drawing') {
    hudModeTag.textContent = '✏️ DRAWING MODE';
    hudModeTag.style.color = '#06b6d4';
    hudModeTag.style.borderColor = '#06b6d4';
    btnToggleMode.textContent = '🎭 Switch to Filter Mode';
  } else {
    hudModeTag.textContent = '🎭 FACE FILTER MODE';
    hudModeTag.style.color = '#ec4899';
    hudModeTag.style.borderColor = '#ec4899';
    btnToggleMode.textContent = '✏️ Switch to Drawing Mode';
  }
}

function toggleAppMode() {
  setAppMode(state.mode === 'drawing' ? 'filter' : 'drawing');
}

function switchNavTab(tabName) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById(`tab-${tabName}`).classList.add('active');

  const drawingPanel = document.getElementById('drawing-panel');
  const subFilters = document.getElementById('sub-filters');
  const subCreator = document.getElementById('sub-creator');

  if (tabName === 'drawing') {
    setAppMode('drawing');
    drawingPanel.style.display = 'flex';
    subFilters.style.display = 'block';
    subCreator.style.display = 'none';
  } else if (tabName === 'filter') {
    setAppMode('filter');
    subFilters.style.display = 'block';
    subCreator.style.display = 'none';
  } else if (tabName === 'creator') {
    subFilters.style.display = 'none';
    subCreator.style.display = 'block';
  }
}

// ── Event Listeners ─────────────────────────────────────────────────────────
function setupEventListeners() {
  // Camera toggles
  btnToggleCam.addEventListener('click', () => {
    state.cameraActive ? stopCamera() : startCamera();
  });
  btnStartHero.addEventListener('click', startCamera);
  btnPauseCam.addEventListener('click', togglePause);
  btnToggleMode.addEventListener('click', toggleAppMode);

  // Snapshot & Download
  btnSnapshot.addEventListener('click', takeSnapshot);
  btnDownloadArt.addEventListener('click', downloadArtworkOnly);

  // Undo / Redo / Clear / Eraser
  btnUndo.addEventListener('click', undoStroke);
  btnRedo.addEventListener('click', redoStroke);
  btnClear.addEventListener('click', clearCanvas);
  btnEraser.addEventListener('click', () => {
    state.eraserActive = !state.eraserActive;
    btnEraser.classList.toggle('active', state.eraserActive);
  });

  // Nav Tabs
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => switchNavTab(tab.dataset.tab));
  });

  // Brushes
  document.querySelectorAll('.brush-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.brush-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeBrush = btn.dataset.brush;
      state.eraserActive = false;
      btnEraser.classList.remove('active');
    });
  });

  // Color Swatches
  document.querySelectorAll('.color-swatch').forEach(swatch => {
    swatch.addEventListener('click', () => {
      document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
      swatch.classList.add('active');
      state.brushColor = swatch.dataset.color;
      updateSizePreview();
    });
  });

  customColorInput.addEventListener('input', (e) => {
    state.brushColor = e.target.value;
    document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
    updateSizePreview();
  });

  // Sliders
  brushSizeSlider.addEventListener('input', (e) => {
    state.brushSize = parseInt(e.target.value);
    sizeValLabel.textContent = `${state.brushSize}px`;
    updateSizePreview();
  });

  glowSlider.addEventListener('input', (e) => {
    state.glowIntensity = parseInt(e.target.value);
    glowValLabel.textContent = `${state.glowIntensity}%`;
  });

  filterIntensitySlider.addEventListener('input', (e) => {
    state.filterIntensity = parseInt(e.target.value) / 100;
    filterIntensityVal.textContent = `${e.target.value}%`;
  });

  // Built-in Filter Cards
  document.querySelectorAll('.filter-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.filter-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      state.activeFilter = card.dataset.filter;
      state.activeCustomFilter = null;
      setAppMode('filter');
    });
  });

  // Custom Filter Studio
  btnConvertFilter.addEventListener('click', convertDrawingToFilter);

  // Checkboxes
  chkLandmarks.addEventListener('change', (e) => state.showLandmarks = e.target.checked);
  chkShapes.addEventListener('change', (e) => state.autoSnapShapes = e.target.checked);
  chkMirror.addEventListener('change', (e) => state.mirror = e.target.checked);

  // Modal
  btnGuide.addEventListener('click', () => modalGuide.classList.add('open'));
  btnCloseModal.addEventListener('click', () => modalGuide.classList.remove('open'));
  btnGotIt.addEventListener('click', () => modalGuide.classList.remove('open'));
  modalGuide.addEventListener('click', (e) => {
    if (e.target === modalGuide) modalGuide.classList.remove('open');
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    if (e.code === 'Space') {
      e.preventDefault();
      state.cameraActive ? stopCamera() : startCamera();
    } else if (e.key === 'p' || e.key === 'P') {
      togglePause();
    } else if (e.code === 'Tab') {
      e.preventDefault();
      toggleAppMode();
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      clearCanvas();
    } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
      e.preventDefault();
      undoStroke();
    } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.shiftKey && e.key === 'Z'))) {
      e.preventDefault();
      redoStroke();
    } else if (e.key === 'e' || e.key === 'E') {
      state.eraserActive = !state.eraserActive;
      btnEraser.classList.toggle('active', state.eraserActive);
    } else if (e.key === '?' || e.key === 'F1') {
      modalGuide.classList.toggle('open');
    }
  });

  window.addEventListener('resize', () => {
    if (video.videoWidth) {
      stageCanvas.width = video.videoWidth;
      stageCanvas.height = video.videoHeight;
    }
  });
}

function updateSizePreview() {
  sizePreviewDot.style.width = `${Math.min(32, state.brushSize)}px`;
  sizePreviewDot.style.height = `${Math.min(32, state.brushSize)}px`;
  sizePreviewDot.style.backgroundColor = state.brushColor;
  sizePreviewDot.style.boxShadow = `0 0 10px ${state.brushColor}`;
}

// ── Application Initialization ──────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initMediaPipe();
  setupEventListeners();
  loadCustomFiltersFromStorage();
  updateSizePreview();
});
