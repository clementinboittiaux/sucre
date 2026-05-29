// Interactive 3D viewer of the SUCRe optimization.
//
// Loads `scene.json` (written by `place_cameras.py --export-web`) and rebuilds, in the
// browser, the same scene `place_cameras.py` builds in Blender: a textured image plane
// and frustum per view camera, the moving light, and the unprojected point cloud.
//
// The timeline scrubs the `numFrames` optimization steps; a separate button swaps the
// view between the reconstructions and the projected illumination at the same step.
// Each per-step image sequence is delivered as a short-GOP H.264 .mp4 (one per (view,
// mode), one for the point cloud color sequence per mode, one for the light pattern);
// the videos are preloaded in full before the scene appears, so playback and scrubbing
// are smooth and never block on the network.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

// One DRACOLoader is shared by every .drc asset; the wasm decoder is fetched
// once and reused. The path matches the three.js version pinned in index.html.
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('https://unpkg.com/three@0.169.0/examples/jsm/libs/draco/');
dracoLoader.preload();

// Fetch a binary asset with byte-accurate progress, reported through the
// shared assetProgress map (see `updateProgressUi`).
async function fetchBytesWithProgress(url, progressKey, assetProgress, updateUi) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) {
      assetProgress.set(progressKey, received / total);
      updateUi();
    }
  }
  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { buffer.set(c, offset); offset += c.length; }
  assetProgress.set(progressKey, 1);
  updateUi();
  return buffer.buffer;
}

// Decode a Draco buffer (mesh or point cloud) into a THREE.BufferGeometry.
// Wraps DRACOLoader (which uses a worker pool) in a Promise. With no extra
// args, uses three.js' default attribute map (POSITION->position etc.); pass
// `attributeIDs` + `attributeTypes` to read by Draco unique_id, e.g. to pull a
// uint16 generic attribute into the geometry's `uv` slot.
function decodeDraco(buffer, attributeIDs, attributeTypes) {
  return new Promise((resolve, reject) => {
    try {
      dracoLoader.decodeDracoFile(
        buffer, (geometry) => resolve(geometry),
        attributeIDs, attributeTypes,
      );
    } catch (err) { reject(err); }
  });
}

const status = document.getElementById('status');
const loadingMsg = document.getElementById('loading-msg');
const progressFill = document.getElementById('progress-fill');

main().catch((err) => {
  console.error(err);
  loadingMsg.textContent = 'Error: ' + err.message;
});

// Procedural "clay" matcap: a sphere shaded with a hemispheric ambient, a tight
// upper-left key highlight, and a soft rim, then tinted. Drawn into a canvas
// and uploaded as a THREE.CanvasTexture so no extra asset has to ship.
//
// Swap MATCAP_TINT to retint. Some pleasant alternatives:
//   warm clay   [0.92, 0.85, 0.74]
//   cool stone  [0.82, 0.86, 0.92]   (default — neutral with a soft blue cast)
//   sage green  [0.82, 0.90, 0.82]
//   lavender    [0.88, 0.82, 0.92]
//   sea foam    [0.78, 0.92, 0.88]
const MATCAP_TINT = [0.82, 0.86, 0.92];

function makeClayMatcap(size = 256) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  const r = size / 2;
  const [tr, tg, tb] = MATCAP_TINT;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 4;
      img.data[idx + 3] = 255;
      const nx = (x - r + 0.5) / r;
      const ny = -((y - r + 0.5) / r);    // image Y goes down; matcap +Y is up
      const d2 = nx * nx + ny * ny;
      if (d2 > 1) {                       // outside the sphere = white background
        img.data[idx] = img.data[idx + 1] = img.data[idx + 2] = 255;
        continue;
      }
      const nz = Math.sqrt(1 - d2);
      // Matte clay: a soft vertical gradient plus a broad, dim key-direction
      // diffuse, all held to a low ceiling. The brightness cap is what removes
      // the glossy look — without it the top of the sphere peaks near white and
      // reads as a specular hotspot.
      const hemi = 0.46 + 0.10 * ny;      // gentle, low-contrast top-to-bottom
      const key  = Math.max(0, nx * -0.4 + ny * 0.5 + nz * 0.77) * 0.16;
      const v = Math.min(0.72, hemi + key);
      img.data[idx]     = Math.round(v * tr * 255);
      img.data[idx + 1] = Math.round(v * tg * 255);
      img.data[idx + 2] = Math.round(v * tb * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

async function main() {
  const scene_data = await fetch('scene.json').then((r) => {
    if (!r.ok) throw new Error('could not load scene.json');
    return r.json();
  });
  const fps = scene_data.fps || 24;
  const numFrames = scene_data.numFrames;

  // --------------------------------------------------------------------- //
  // Collect every distinct video URL referenced by the scene.
  // --------------------------------------------------------------------- //
  const videoUrls = new Set();
  for (const v of scene_data.views) {
    videoUrls.add(v.reconstructionVideo);
    if (v.vignettingVideo) videoUrls.add(v.vignettingVideo);
  }
  if (scene_data.light) videoUrls.add(scene_data.light.patternVideo);
  if (scene_data.pointCloud) {
    videoUrls.add(scene_data.pointCloud.rgbVideo);
    if (scene_data.pointCloud.vignettingVideo) videoUrls.add(scene_data.pointCloud.vignettingVideo);
  }

  // --------------------------------------------------------------------- //
  // Preload every video in full, updating a progress bar from `buffered`.
  // --------------------------------------------------------------------- //
  const videos = new Map();
  for (const url of videoUrls) {
    const el = document.createElement('video');
    el.src = url;
    el.muted = true;
    el.playsInline = true;
    el.preload = 'auto';
    el.crossOrigin = 'anonymous';
    el.load();
    videos.set(url, { element: el, texture: null });
  }

  // Per-asset progress (videos + Draco mesh + Draco point cloud), averaged for the UI bar.
  const assetProgress = new Map();
  for (const url of videoUrls) assetProgress.set(url, 0);
  if (scene_data.mesh) assetProgress.set('mesh', 0);
  if (scene_data.pointCloud) assetProgress.set('pointcloud', 0);

  function updateProgressUi() {
    let sum = 0;
    for (const p of assetProgress.values()) sum += p;
    const pct = (sum / assetProgress.size) * 100;
    progressFill.style.width = `${pct.toFixed(1)}%`;
    loadingMsg.textContent = `Loading ${pct.toFixed(0)}%`;
  }

  const videosReady = new Promise((resolve, reject) => {
    const update = () => {
      for (const [url, { element }] of videos) {
        if (element.error) {
          reject(new Error(`failed to load ${element.src}`));
          return;
        }
        let p = 0;
        if (element.duration && element.buffered.length > 0) {
          p = element.buffered.end(element.buffered.length - 1) / element.duration;
        }
        if (element.readyState >= 4) p = 1;
        assetProgress.set(url, p);
      }
      updateProgressUi();
      let allReady = true;
      for (const { element } of videos.values()) {
        if (element.readyState < 4) { allReady = false; break; }
      }
      if (allReady) {
        clearInterval(interval);
        resolve();
      }
    };
    const interval = setInterval(update, 200);
    update();
  });

  // Fetch the Draco-compressed mesh and point cloud with byte-accurate progress,
  // then decode each into a BufferGeometry (DRACOLoader runs in a worker pool).
  let meshGeometry = null;
  const meshReady = scene_data.mesh ? (async () => {
    const buf = await fetchBytesWithProgress(scene_data.mesh, 'mesh',
                                             assetProgress, updateProgressUi);
    meshGeometry = await decodeDraco(buf);
    meshGeometry.computeVertexNormals();
  })() : Promise.resolve();

  let pointCloudGeometry = null;
  const pointCloudReady = scene_data.pointCloud ? (async () => {
    const buf = await fetchBytesWithProgress(scene_data.pointCloud.data, 'pointcloud',
                                             assetProgress, updateProgressUi);
    // UVs were encoded as a uint16 generic attribute at unique_id=1 (see
    // sucre/export_web.py). Read it into the geometry's `uv` slot and mark it
    // normalized so the shader gets the original float values in [0, 1].
    pointCloudGeometry = await decodeDraco(buf,
      { position: 0, uv: 1 },
      { position: 'Float32Array', uv: 'Uint16Array' });
    if (pointCloudGeometry.attributes.uv) {
      pointCloudGeometry.attributes.uv.normalized = true;
    }
  })() : Promise.resolve();

  await Promise.all([videosReady, meshReady, pointCloudReady]);

  // The videos are buffered; wrap each one in a VideoTexture for the GPU.
  for (const entry of videos.values()) {
    entry.texture = new THREE.VideoTexture(entry.element);
    entry.texture.colorSpace = THREE.SRGBColorSpace;
  }

  // --------------------------------------------------------------------- //
  // Renderer, scene, camera, controls.
  // --------------------------------------------------------------------- //
  const app = document.getElementById('app');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(app.clientWidth, app.clientHeight);
  app.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  // Lights for the surface mesh's MeshStandardMaterial (everything else is unlit).
  scene.add(new THREE.HemisphereLight(0xffffff, 0x555555, 2.0));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
  keyLight.position.set(-1.5, -2, 2.5);
  scene.add(keyLight);

  const camera = new THREE.PerspectiveCamera(50, app.clientWidth / app.clientHeight, 0.01, 1e4);
  camera.up.set(0, 0, 1);   // COLMAP / Blender world is Z-up

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // --------------------------------------------------------------------- //
  // Geometry helpers: a textured quad and a frustum wireframe from world points.
  // --------------------------------------------------------------------- //
  const QUAD_UV = [1, 1, 1, 0, 0, 0, 0, 1];   // corners are ordered TR, BR, BL, TL
  const QUAD_INDEX = [0, 1, 2, 0, 2, 3];

  function makeQuad(corners, texture) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(corners.flat(), 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(QUAD_UV, 2));
    geometry.setIndex(QUAD_INDEX);
    const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide,
                                                   toneMapped: false });
    return new THREE.Mesh(geometry, material);
  }

  function frustumPoints(apex, corners) {   // 8 segments: apex->corners and the rim
    const pts = [];
    for (const c of corners) pts.push(apex, c);
    for (let i = 0; i < 4; i++) pts.push(corners[i], corners[(i + 1) % 4]);
    return pts.flat();
  }

  function makeFrustum(apex, corners, color) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position',
      new THREE.Float32BufferAttribute(frustumPoints(apex, corners), 3));
    return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color }));
  }

  function setQuadCorners(quad, corners) {
    quad.geometry.attributes.position.array.set(corners.flat());
    quad.geometry.attributes.position.needsUpdate = true;
  }

  function setFrustum(frustum, apex, corners) {
    frustum.geometry.attributes.position.array.set(frustumPoints(apex, corners));
    frustum.geometry.attributes.position.needsUpdate = true;
  }

  // --------------------------------------------------------------------- //
  // View cameras: a textured image plane and a frustum each. The view holds the
  // VideoTextures and <video> elements for both phases, so phase switches are a
  // cheap material.map swap.
  // --------------------------------------------------------------------- //
  const views = [];
  for (const v of scene_data.views) {
    const reconTex = videos.get(v.reconstructionVideo).texture;
    const reconVideo = videos.get(v.reconstructionVideo).element;
    const vignTex = v.vignettingVideo ? videos.get(v.vignettingVideo).texture : null;
    const vignVideo = v.vignettingVideo ? videos.get(v.vignettingVideo).element : null;
    const quad = makeQuad(v.corners, reconTex);
    scene.add(quad);
    scene.add(makeFrustum(v.apex, v.corners, v.isTarget ? 0xffcc33 : 0x6699ff));
    views.push({ data: v, quad, reconTex, vignTex, reconVideo, vignVideo });
  }

  // --------------------------------------------------------------------- //
  // Light: camera-like frustum + pattern plane, animated per optimization step.
  // The pattern video is the same for both phases (it just replays).
  // --------------------------------------------------------------------- //
  let light = null;
  if (scene_data.light) {
    const data = scene_data.light;
    const patternTex = videos.get(data.patternVideo).texture;
    const patternVideo = videos.get(data.patternVideo).element;
    const quad = makeQuad(data.corners[0], patternTex);
    const frustum = makeFrustum(data.apex[0], data.corners[0], 0xff5533);
    for (const mat of [quad.material, frustum.material]) {
      mat.transparent = true;
      mat.opacity = 0.5;
    }
    scene.add(quad);
    scene.add(frustum);
    light = { data, quad, frustum, video: patternVideo };
  }

  // --------------------------------------------------------------------- //
  // Point cloud: a shader samples the per-step color video at each point's UV.
  // The Draco-decoded BufferGeometry already carries `position` and `uv`.
  // --------------------------------------------------------------------- //
  let pointCloud = null;
  if (scene_data.pointCloud) {
    const pc = scene_data.pointCloud;
    const geometry = pointCloudGeometry;
    const rgbTex = videos.get(pc.rgbVideo).texture;
    const rgbVideo = videos.get(pc.rgbVideo).element;
    const vignTex = pc.vignettingVideo ? videos.get(pc.vignettingVideo).texture : null;
    const vignVideo = pc.vignettingVideo ? videos.get(pc.vignettingVideo).element : null;
    const material = new THREE.ShaderMaterial({
      uniforms: { colorMap: { value: rgbTex }, pointScale: { value: 13.5 } },
      vertexShader: `
        uniform sampler2D colorMap;
        uniform float pointScale;
        varying vec3 vColor;
        void main() {
          vColor = texture2D(colorMap, uv).rgb;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = max(1.0, pointScale / -mv.z);
          gl_Position = projectionMatrix * mv;
          // Depth bias: shift the depth value 5 cm toward the camera so points
          // up to 5 cm behind the mesh still pass the depth test, but anything
          // further back is still correctly occluded.
          vec4 clipBiased = projectionMatrix * vec4(mv.xy, mv.z + 0.05, mv.w);
          gl_Position.z = clipBiased.z * (gl_Position.w / clipBiased.w);
        }`,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          vec2 d = gl_PointCoord - vec2(0.5);
          if (dot(d, d) > 0.25) discard;            // round points
          gl_FragColor = vec4(vColor, 1.0);
        }`,
    });
    scene.add(new THREE.Points(geometry, material));
    pointCloud = { data: pc, material, rgbTex, vignTex, rgbVideo, vignVideo };
  }

  // --------------------------------------------------------------------- //
  // Surface mesh of the reconstruction. A MeshStandardMaterial lit by the
  // scene's hemisphere + directional lights so its shape reads; everything else
  // in the scene uses unlit materials (MeshBasic / LineBasic / ShaderMaterial).
  // --------------------------------------------------------------------- //
  if (meshGeometry) {
    const mesh = new THREE.Mesh(meshGeometry,
      new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.9, metalness: 0.0,
                                       side: THREE.DoubleSide }));
    scene.add(mesh);
  }

  // --------------------------------------------------------------------- //
  // Frame the whole scene.
  // --------------------------------------------------------------------- //
  const box = new THREE.Box3().setFromObject(scene);
  const radius = box.getSize(new THREE.Vector3()).length() * 0.5 || 1;
  controls.target.set(1.163, 6.009, 4.589);
  camera.position.set(-4.224, -8.402, 10.916);
  camera.near = radius / 100;
  camera.far = radius * 100;
  camera.updateProjectionMatrix();

  // DEBUG: live viewpoint readout — remove after choosing the start view.
  const vpReadout = document.createElement('div');
  vpReadout.style.cssText = 'position:fixed;top:8px;left:8px;z-index:9999;' +
    'font:12px/1.4 monospace;background:rgba(0,0,0,.75);color:#0f0;' +
    'padding:6px 8px;border-radius:4px;white-space:pre;pointer-events:none;';
  document.body.appendChild(vpReadout);
  const vpFmt = v => `${v.x.toFixed(3)}, ${v.y.toFixed(3)}, ${v.z.toFixed(3)}`;
  function showViewpoint() {
    const snippet = `camera.position.set(${vpFmt(camera.position)});\n`
                  + `controls.target.set(${vpFmt(controls.target)});`;
    vpReadout.textContent = snippet;
    return snippet;
  }
  showViewpoint();
  let vpTimer = null;
  controls.addEventListener('change', () => {
    showViewpoint();
    clearTimeout(vpTimer);
    vpTimer = setTimeout(() => console.log('[viewpoint]\n' + showViewpoint()), 400);
  });
  // END DEBUG

  // --------------------------------------------------------------------- //
  // Timeline + view-mode toggle. The slider scrubs optimization steps; the
  // view-toggle button switches what the cameras and point cloud show at the
  // current step (reconstruction <-> projected illumination).
  // --------------------------------------------------------------------- //
  const stepInput = document.getElementById('frame');
  const stepLabel = document.getElementById('phase');
  const viewToggleButton = document.getElementById('view-toggle');
  stepInput.max = numFrames - 1;

  let viewMode = 0;   // 0 = reconstruction, 1 = illumination

  function activeVideos(mode) {
    const out = [];
    for (const view of views) out.push((mode === 1 && view.vignVideo) ? view.vignVideo
                                                                      : view.reconVideo);
    if (pointCloud) out.push((mode === 1 && pointCloud.vignVideo) ? pointCloud.vignVideo
                                                                  : pointCloud.rgbVideo);
    if (light) out.push(light.video);
    return out;
  }

  function bindMode(mode) {
    for (const view of views) {
      view.quad.material.map = (mode === 1 && view.vignTex) ? view.vignTex : view.reconTex;
    }
    if (pointCloud) {
      pointCloud.material.uniforms.colorMap.value = (mode === 1 && pointCloud.vignTex)
        ? pointCloud.vignTex : pointCloud.rgbTex;
    }
    viewToggleButton.textContent = mode === 0 ? 'Show illumination' : 'Show reconstruction';
  }

  function pauseAll() {
    for (const { element } of videos.values()) element.pause();
  }

  function seekTo(step) {
    const t = step / fps;
    for (const v of activeVideos(viewMode)) v.currentTime = t;
  }

  function updateLightForStep(step) {
    if (!light) return;
    setQuadCorners(light.quad, light.data.corners[step]);
    setFrustum(light.frustum, light.data.apex[step], light.data.corners[step]);
  }

  function updateLabel(step) {
    const name = viewMode ? 'illumination' : 'reconstruction';
    stepLabel.textContent = `${name} — step ${step} / ${numFrames - 1}`;
  }

  function gotoStep(s) {
    const step = ((s % numFrames) + numFrames) % numFrames;
    stepInput.value = step;
    updateLightForStep(step);
    updateLabel(step);
    if (!playing) seekTo(step);
  }

  // --------------------------------------------------------------------- //
  // UI: scrub slider, play / pause, and the view-mode toggle.
  // --------------------------------------------------------------------- //
  const playButton = document.getElementById('play');
  let playing = false;

  function startPlayback() {
    pauseAll();
    const step = parseInt(stepInput.value, 10);
    const t = step / fps;
    for (const v of activeVideos(viewMode)) {
      v.currentTime = t;
      v.play().catch(() => {});
    }
  }

  stepInput.addEventListener('input', () => {
    if (playing) { playing = false; pauseAll(); playButton.innerHTML = '&#9654;'; }
    gotoStep(parseInt(stepInput.value, 10));
  });

  playButton.addEventListener('click', () => {
    playing = !playing;
    playButton.innerHTML = playing ? '&#10073;&#10073;' : '&#9654;';
    if (playing) startPlayback();
    else pauseAll();
  });

  viewToggleButton.addEventListener('click', () => {
    viewMode = viewMode === 0 ? 1 : 0;
    bindMode(viewMode);
    const step = parseInt(stepInput.value, 10);
    if (playing) startPlayback();
    else seekTo(step);
    updateLabel(step);
  });

  window.addEventListener('resize', () => {
    renderer.setSize(app.clientWidth, app.clientHeight);
    camera.aspect = app.clientWidth / app.clientHeight;
    camera.updateProjectionMatrix();
  });

  bindMode(viewMode);
  gotoStep(0);
  status.hidden = true;
  document.getElementById('ui').hidden = false;
  if (scene_data.phased) viewToggleButton.hidden = false;

  // Start playback automatically once the preload completes.
  playing = true;
  playButton.innerHTML = '&#10073;&#10073;';
  startPlayback();

  renderer.setAnimationLoop(() => {
    if (playing) {
      // Use the first view's active video as the master clock for the timeline UI.
      const lead = (viewMode === 1 && views[0].vignVideo) ? views[0].vignVideo
                                                          : views[0].reconVideo;
      if (lead.ended) {
        // Loop back to the start of the current mode.
        for (const v of activeVideos(viewMode)) { v.currentTime = 0; v.play().catch(() => {}); }
      }
      const step = Math.min(numFrames - 1, Math.floor(lead.currentTime * fps));
      stepInput.value = step;
      updateLightForStep(step);
      updateLabel(step);
    }
    controls.update();
    renderer.render(scene, camera);
  });
}
