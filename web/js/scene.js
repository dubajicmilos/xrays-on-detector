/*! The Game of Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Three.js scene for the six-circle diffractometer.
 *
 * Geometry is built once and only transforms are touched per frame. The
 * goniometer is a nested hierarchy of Object3D groups (mu carries eta carries
 * chi carries phi), each with its matrix set straight from the corresponding
 * 3x3 circle matrix, so the scene graph does the composition rather than the
 * code repeating it.
 *
 * Lab frame: +x vertical (up), +y along the beam, +z horizontal. The camera up
 * vector is set to +x to match.
 */
import * as THREE from "../lib/three.module.js";

import { detectorMatrix, sampleMatrix } from "./physics.js";
import { whiteboardTexture } from "./credit.js";

const COL = {
  mu: 0xe85d75,
  eta: 0xf0be46,
  chi: 0x5fc88c,
  phi: 0x5aa0ff,
  detector: 0x9aa4b8,
  beam: 0xffe896,
  ray: 0x78e6ff,
  miss: 0x8296b9,
  block: 0xeb6e5f,
  sample: 0xdce1f0,
  surface: 0x8ca0d0,
  a: 0xff6e6e,
  b: 0x8ceb8c,
  c: 0x82afff,
  labX: 0xeb7890,
  labY: 0xf0cd6e,
  labZ: 0x78c8ff,
  grid: 0x2b3a5c,
};

/** Matrix4 from a row-major 3x3 rotation. */
function mat4From3(m) {
  const M = new THREE.Matrix4();
  M.set(
    m[0][0],
    m[0][1],
    m[0][2],
    0,
    m[1][0],
    m[1][1],
    m[1][2],
    0,
    m[2][0],
    m[2][1],
    m[2][2],
    0,
    0,
    0,
    0,
    1,
  );
  return M;
}

/** A text label that always faces the camera. */
function makeLabel(text, color = "#dfe6f5", size = 44) {
  const pad = 8;
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  ctx.font = `600 ${size}px Segoe UI, system-ui, sans-serif`;
  c.width = Math.ceil(ctx.measureText(text).width) + pad * 2;
  c.height = size + pad * 2;
  const ctx2 = c.getContext("2d");
  ctx2.font = `600 ${size}px Segoe UI, system-ui, sans-serif`;
  ctx2.fillStyle = color;
  ctx2.textBaseline = "middle";
  ctx2.shadowColor = "rgba(0,0,0,0.85)";
  ctx2.shadowBlur = 8;
  ctx2.fillText(text, pad, c.height / 2);

  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    }),
  );
  sprite.renderOrder = 999;
  sprite.userData.aspect = c.width / c.height;
  return sprite;
}

/** Orbit / pan / zoom, mouse and touch, written here so there is no addon. */
class Orbit {
  constructor(dom, onChange) {
    this.dom = dom;
    this.onChange = onChange;
    this.target = new THREE.Vector3(0, 0, 0);
    this.distance = 900;
    this.azimuth = 40;
    this.elevation = 22;
    this.minDistance = 40;
    this.maxDistance = 20000;

    this._pointers = new Map();
    this._last = null;
    this._pinch = 0;

    dom.style.touchAction = "none";
    dom.addEventListener("pointerdown", (e) => this._down(e));
    dom.addEventListener("pointermove", (e) => this._move(e));
    dom.addEventListener("pointerup", (e) => this._up(e));
    dom.addEventListener("pointercancel", (e) => this._up(e));
    dom.addEventListener("wheel", (e) => this._wheel(e), { passive: false });
    dom.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  _down(e) {
    this.dom.setPointerCapture(e.pointerId);
    this._pointers.set(e.pointerId, {
      x: e.clientX,
      y: e.clientY,
      button: e.button,
    });
    this._last = { x: e.clientX, y: e.clientY, button: e.button };
    if (this._pointers.size === 2) this._pinch = this._pinchDistance();
  }

  _pinchDistance() {
    const [p, q] = [...this._pointers.values()];
    return Math.hypot(p.x - q.x, p.y - q.y);
  }

  _move(e) {
    if (!this._pointers.has(e.pointerId)) return;
    const prev = this._pointers.get(e.pointerId);
    this._pointers.set(e.pointerId, { ...prev, x: e.clientX, y: e.clientY });

    if (this._pointers.size === 2) {
      const d = this._pinchDistance();
      if (this._pinch > 0) this.zoom(this._pinch / d);
      this._pinch = d;
      return;
    }
    if (!this._last) return;
    const dx = e.clientX - this._last.x;
    const dy = e.clientY - this._last.y;
    this._last = { x: e.clientX, y: e.clientY, button: this._last.button };

    if (this._last.button === 2 || e.shiftKey) this.pan(dx, dy);
    else this.orbit(-dx * 0.35, dy * 0.35);
  }

  _up(e) {
    this._pointers.delete(e.pointerId);
    if (this._pointers.size < 2) this._pinch = 0;
    if (this._pointers.size === 0) this._last = null;
  }

  _wheel(e) {
    e.preventDefault();
    this.zoom(e.deltaY > 0 ? 1 / 0.88 : 0.88);
  }

  orbit(daz, dele) {
    this.azimuth = (this.azimuth + daz) % 360;
    this.elevation = Math.max(-89, Math.min(89, this.elevation + dele));
    this.onChange();
  }

  zoom(f) {
    this.distance = Math.max(
      this.minDistance,
      Math.min(this.maxDistance, this.distance * f),
    );
    this.onChange();
  }

  pan(dx, dy) {
    const { right, up } = this.basis();
    const s = this.distance * 0.0015;
    this.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
    this.onChange();
  }

  eye() {
    const el = (this.elevation * Math.PI) / 180;
    const az = (this.azimuth * Math.PI) / 180;
    return new THREE.Vector3(
      Math.sin(el),
      Math.cos(el) * Math.cos(az),
      Math.cos(el) * Math.sin(az),
    )
      .multiplyScalar(this.distance)
      .add(this.target);
  }

  basis() {
    const eye = this.eye();
    const fwd = this.target.clone().sub(eye).normalize();
    const upRef = new THREE.Vector3(1, 0, 0);
    let right = fwd.clone().cross(upRef);
    if (right.lengthSq() < 1e-12) right = new THREE.Vector3(0, 0, 1);
    right.normalize();
    const up = right.clone().cross(fwd).normalize();
    return { eye, right, up, fwd };
  }

  apply(camera) {
    const { eye, up } = this.basis();
    camera.position.copy(eye);
    camera.up.copy(up);
    camera.lookAt(this.target);
  }
}

// ---------------------------------------------------------------------------

export class InstrumentScene {
  /**
   * Physical sizes in millimetres. The goniometer is real hardware, so it has
   * a fixed size rather than one derived from the detector distance: only the
   * camera adapts when the detector moves.
   */
  static DIM = {
    ringMu: 95,
    ringEta: 76,
    ringChi: 58,
    ringPhi: 40,
    tube: 3.2,
    sample: 30,
    surface: 66,
    beam: 300,
    floorY: -125,
    floorSpan: 900,
    gizmo: 82,
    gizmoY: -55,
    gizmoZ: 178,
    boardLen: 560,
    boardHeight: 370,
    boardThick: 14,
    boardClear: 60,
    boardMinZ: 240,
    boardY: 20,
  };

  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0e18);
    this.scene.fog = new THREE.Fog(0x0b0e18, 2200, 6000);

    this.camera = new THREE.PerspectiveCamera(42, 1, 1, 30000);
    this.controls = new Orbit(canvas, () => {
      this.dirty = true;
    });
    // Look from upstream so the detector's face, not its casing, is toward us.
    this.controls.azimuth = 138;
    this.controls.elevation = 19;
    this.controls.distance = 760;
    this.dirty = true;
    this.scale = 200;

    this._lights();
    this._build();
    this.resize();
  }

  _lights() {
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(1.0, -0.6, 0.8);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.9);
    fill.position.set(-0.8, 0.5, -0.6);
    this.scene.add(fill);
  }

  _build() {
    const D = InstrumentScene.DIM;

    // -- floor -------------------------------------------------------------
    this.floor = new THREE.GridHelper(D.floorSpan, 18, COL.grid, COL.grid);
    // GridHelper lies in the XZ plane with +y up; our vertical is +x.
    this.floor.rotation.z = Math.PI / 2;
    this.floor.position.set(D.floorY, 0, 0);
    this.floor.material.transparent = true;
    this.floor.material.opacity = 0.45;
    this.scene.add(this.floor);

    // -- whiteboard --------------------------------------------------------
    // A lab whiteboard standing on the floor alongside the beam, carrying the
    // equations, the credit and the two wanted posters. Its plane holds the
    // beam direction and the vertical, so its face is normal to z: look across
    // the beam and you read it. Fixed size, like the goniometer and the floor;
    // only its z follows the detector, in _placeBoard.
    //
    // A BoxGeometry rather than a plane so it has thickness and reads as a
    // plate from any angle. Local axes after the rotation below: local +x is
    // -y in the lab (upstream), local +y is the lab vertical, local +z is lab
    // +z. Material index 4 is that local +z face, which is the printed side
    // and the one the opening camera looks at.
    const board = new THREE.BoxGeometry(
      D.boardLen,
      D.boardHeight,
      D.boardThick,
    );
    const edge = new THREE.MeshBasicMaterial({ color: 0x2b3348 });
    this.boardTexture = new THREE.CanvasTexture(
      // The posters arrive over the network, so the board is redrawn once they
      // do and the texture pushed up again.
      whiteboardTexture(D.boardLen, D.boardHeight, () => {
        this.boardTexture.needsUpdate = true;
        this.dirty = true;
      }),
    );
    // Without this the canvas is taken as linear and converted on output, and
    // the board comes out several stops brighter than the colours it was drawn
    // with: a pale wall instead of dark anodised aluminium.
    this.boardTexture.colorSpace = THREE.SRGBColorSpace;
    this.boardTexture.anisotropy =
      this.renderer.capabilities.getMaxAnisotropy();
    this.board = new THREE.Mesh(board, [
      edge,
      edge,
      edge,
      edge,
      new THREE.MeshBasicMaterial({ map: this.boardTexture }),
      edge,
    ]);
    this.board.rotation.z = -Math.PI / 2;
    this.board.position.set(
      D.floorY + D.boardHeight / 2, // stands on the floor
      D.boardY,
      -D.boardMinZ, // replaced on the first update
    );
    this.scene.add(this.board);

    // -- goniometer hierarchy ---------------------------------------------
    this.gMu = new THREE.Group();
    this.gEta = new THREE.Group();
    this.gChi = new THREE.Group();
    this.gPhi = new THREE.Group();
    for (const g of [this.gMu, this.gEta, this.gChi, this.gPhi])
      g.matrixAutoUpdate = false;
    this.scene.add(this.gMu);
    this.gMu.add(this.gEta);
    this.gEta.add(this.gChi);
    this.gChi.add(this.gPhi);

    const ring = (radius, colour, axis, arc = Math.PI * 2) => {
      const geo = new THREE.TorusGeometry(radius, D.tube, 12, 96, arc);
      const mat = new THREE.MeshStandardMaterial({
        color: colour,
        roughness: 0.45,
        metalness: 0.35,
      });
      const m = new THREE.Mesh(geo, mat);
      // Torus normal is +z by default.
      if (axis === "x") m.rotateY(Math.PI / 2);
      else if (axis === "y") m.rotateX(Math.PI / 2);
      return m;
    };

    this.ringMu = ring(D.ringMu, COL.mu, "x");
    this.ringEta = ring(D.ringEta, COL.eta, "z");
    this.ringChi = ring(D.ringChi, COL.chi, "y", (210 * Math.PI) / 180);
    this.ringPhi = ring(D.ringPhi, COL.phi, "z");
    this.scene.add(this.ringMu); // mu is fixed to the lab
    this.gMu.add(this.ringEta);
    this.gEta.add(this.ringChi);
    this.gChi.add(this.ringPhi);

    this.ringLabels = {};
    for (const [name, host, radius] of [
      ["mu", this.scene, D.ringMu],
      ["omega", this.gMu, D.ringEta],
      ["chi", this.gEta, D.ringChi],
      ["phi", this.gChi, D.ringPhi],
    ]) {
      const key = name === "omega" ? "eta" : name;
      const s = makeLabel(
        name,
        `#${COL[key].toString(16).padStart(6, "0")}`,
        40,
      );
      s.position.set(0, 0, 0);
      s.userData.radius = radius;
      host.add(s);
      this.ringLabels[name] = s;
    }
    // put each label just off the rim of its own ring
    this.ringLabels.mu.position.set(0, 0, D.ringMu * 1.18);
    this.ringLabels.omega.position.set(D.ringEta * 1.18, 0, 0);
    this.ringLabels.chi.position.set(D.ringChi * 1.18, 0, 0);
    this.ringLabels.phi.position.set(D.ringPhi * 1.18, 0, 0);

    // -- sample ------------------------------------------------------------
    // The crystal sits in the phi frame but is oriented by U, so turning the
    // crystal with the rx/ry/rz sliders visibly turns the block.
    this.sample = new THREE.Mesh(
      new THREE.BoxGeometry(D.sample, D.sample * 0.72, D.sample * 0.55),
      new THREE.MeshStandardMaterial({
        color: COL.sample,
        roughness: 0.25,
        metalness: 0.15,
        emissive: 0x223049,
        emissiveIntensity: 0.6,
      }),
    );
    this.sample.matrixAutoUpdate = false;
    this.sampleEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(this.sample.geometry),
      new THREE.LineBasicMaterial({ color: 0x7f8db0 }),
    );
    this.sampleEdges.matrixAutoUpdate = false;
    this.gPhi.add(this.sample, this.sampleEdges);

    this.surface = new THREE.Mesh(
      new THREE.BoxGeometry(D.surface, D.surface * 0.07, D.surface),
      new THREE.MeshStandardMaterial({
        color: COL.surface,
        roughness: 0.35,
        metalness: 0.25,
      }),
    );
    this.surface.matrixAutoUpdate = false;
    this.surface.visible = false;
    this.scene.add(this.surface);

    // -- beam --------------------------------------------------------------
    const beamGeo = new THREE.CylinderGeometry(2.6, 2.6, D.beam, 12);
    this.beam = new THREE.Mesh(
      beamGeo,
      new THREE.MeshBasicMaterial({
        color: COL.beam,
        transparent: true,
        opacity: 0.85,
      }),
    );
    this.beam.position.set(0, -D.beam / 2, 0); // cylinder axis is +y already
    this.scene.add(this.beam);

    // -- detector ----------------------------------------------------------
    this.detGroup = new THREE.Group();
    this.detGroup.matrixAutoUpdate = false;
    this.scene.add(this.detGroup);

    this.detTexCanvas = document.createElement("canvas");
    this.detTexCanvas.width = 8;
    this.detTexCanvas.height = 8;
    this.detTexture = new THREE.CanvasTexture(this.detTexCanvas);
    this.detTexture.minFilter = THREE.LinearMipmapLinearFilter;
    this.detTexture.magFilter = THREE.LinearFilter;
    this.detTexture.generateMipmaps = true;

    this.detFace = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({
        map: this.detTexture,
        side: THREE.DoubleSide,
      }),
    );
    this.detGroup.add(this.detFace);

    this.detCase = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({
        color: COL.detector,
        roughness: 0.6,
        metalness: 0.3,
      }),
    );
    this.detGroup.add(this.detCase);

    this.armLine = this._line(COL.detector, 0.55);
    this.scene.add(this.armLine);

    this.detLabel = makeLabel("detector", "#c8d2ea", 40);
    this.detGroup.add(this.detLabel);

    // -- ray bundles -------------------------------------------------------
    this.rayHit = this._lineSegments(COL.ray, 0.95, true);
    this.rayMiss = this._lineSegments(COL.miss, 0.5);
    this.rayBlock = this._lineSegments(COL.block, 0.6);
    this.directBeam = this._line(COL.beam, 0.3);

    // -- gizmos ------------------------------------------------------------
    this.gizmoLab = this._triad(
      ["x  up", "y  beam", "z"],
      [COL.labX, COL.labY, COL.labZ],
    );
    this.gizmoCry = this._triad(["a", "b", "c"], [COL.a, COL.b, COL.c]);
    // Standing on the floor, one either side of the incoming beam. Their
    // materials ignore the depth buffer so the detector arm can never hide
    // them, which is what made the earlier in-scene version unusable.
    this.gizmoLab.group.position.set(D.floorY, D.gizmoY, -D.gizmoZ);
    this.gizmoCry.group.position.set(D.floorY, D.gizmoY, D.gizmoZ);
    this.scene.add(this.gizmoLab.group, this.gizmoCry.group);
    this.gizmoLab.setDirections([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    for (const [g, text] of [
      [this.gizmoLab, "lab"],
      [this.gizmoCry, "crystal"],
    ]) {
      const s = makeLabel(text, "#a5b2cd", 40);
      s.position.set(-D.gizmo * 0.42, 0, 0);
      g.group.add(s);
      g.title = s;
    }
  }

  _line(colour, opacity) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(6), 3),
    );
    const m = new THREE.Line(
      geo,
      new THREE.LineBasicMaterial({
        color: colour,
        transparent: true,
        opacity,
      }),
    );
    this.scene.add(m);
    return m;
  }

  _lineSegments(colour, opacity, additive = false) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(6), 3),
    );
    const m = new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({
        color: colour,
        transparent: true,
        opacity,
        // Light adds. It also means a ray whose colour has been faded to black
        // contributes nothing at all, instead of painting a black line that is
        // still visible against the panel.
        blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
        depthWrite: !additive,
      }),
    );
    m.frustumCulled = false;
    this.scene.add(m);
    return m;
  }

  /** Three arrows from a common origin, with labels. */
  _triad(names, colours) {
    const group = new THREE.Group();
    const L = InstrumentScene.DIM.gizmo;
    const arrows = [];
    const labels = [];
    names.forEach((name, i) => {
      const arrow = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 0, 0),
        L,
        colours[i],
        L * 0.22,
        L * 0.09,
      );
      arrow.line.material.depthTest = false;
      arrow.line.material.linewidth = 2;
      arrow.cone.material.depthTest = false;
      arrow.line.renderOrder = 900;
      arrow.cone.renderOrder = 900;
      group.add(arrow);
      arrows.push(arrow);
      const s = makeLabel(
        name,
        `#${colours[i].toString(16).padStart(6, "0")}`,
        44,
      );
      group.add(s);
      labels.push(s);
    });
    const setDirections = (dirs) => {
      dirs.forEach((d, i) => {
        const v = new THREE.Vector3(d[0], d[1], d[2]);
        if (v.lengthSq() < 1e-12) return;
        v.normalize();
        arrows[i].setDirection(v);
        labels[i].position.copy(v).multiplyScalar(L * 1.16);
      });
    };
    return { group, arrows, labels, setDirections };
  }

  // -- per-frame update ---------------------------------------------------

  /**
   * @param {object} st
   *   angles      {mu, eta, chi, phi, delta, gamma}
   *   detector    {distance, nFast, nSlow, pixelSize}
   *   frame       {centre, eFast, eSlow, arm} from Detector.frame()
   *   rays        {hit: Float32Array pairs, miss, block}
   *   crystalAxes [[ax,ay,az],[bx,by,bz],[cx,cy,cz]] unit vectors in the lab
   *   surface     null, or {normal: [x,y,z]}
   *   show        {rings, rays, missed, floor, axes}
   */
  update(st) {
    const { mu, eta, chi, phi } = st.angles;

    this.gMu.matrix.copy(mat4From3(sampleMatrix(mu, 0, 0, 0)));
    this.gEta.matrix.copy(mat4From3(sampleMatrix(0, eta, 0, 0)));
    this.gChi.matrix.copy(mat4From3(sampleMatrix(0, 0, chi, 0)));
    this.gPhi.matrix.copy(mat4From3(sampleMatrix(0, 0, 0, phi)));

    // detector: basis (eFast, eSlow, -arm) is right-handed and puts the
    // plane's front face toward the sample. The plane's +y must be +eSlow,
    // not -eSlow: the texture's top row is the largest slow pixel, so with
    // -eSlow every spot was drawn mirrored about the panel's horizontal
    // centre line and no longer sat where its own ray lands.
    const { centre, eFast, eSlow, arm } = st.frame;
    const M = new THREE.Matrix4();
    M.set(
      eFast[0],
      eSlow[0],
      -arm[0],
      centre[0],
      eFast[1],
      eSlow[1],
      -arm[1],
      centre[1],
      eFast[2],
      eSlow[2],
      -arm[2],
      centre[2],
      0,
      0,
      0,
      1,
    );
    this.detGroup.matrix.copy(M);

    const w = st.detector.nFast * st.detector.pixelSize;
    const h = st.detector.nSlow * st.detector.pixelSize;
    this._placeBoard(w);
    this.detFace.scale.set(w, h, 1);
    this.detCase.scale.set(w * 1.05, h * 1.05, 22);
    this.detCase.position.set(0, 0, -13);
    this.detLabel.position.set(-w * 0.4, h * 0.58, 0);

    this._setLine(this.armLine, [0, 0, 0], centre);

    // direct beam through to the detector plane
    const denom = -arm[1];
    if (Math.abs(denom) > 1e-9) {
      const t =
        -(centre[0] * arm[0] + centre[1] * arm[1] + centre[2] * arm[2]) / denom;
      this.directBeam.visible = t > 0;
      if (t > 0) this._setLine(this.directBeam, [0, 0, 0], [0, t, 0]);
    } else this.directBeam.visible = false;

    this._setSegments(this.rayHit, st.rays.hit, st.rays.hitColour);
    this._setSegments(this.rayMiss, st.show.missed ? st.rays.miss : null);
    this._setSegments(this.rayBlock, st.show.missed ? st.rays.block : null);

    if (st.surface) {
      const n = new THREE.Vector3(...st.surface.normal).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        n,
      );
      this.surface.matrix.compose(
        new THREE.Vector3(0, 0, 0),
        q,
        new THREE.Vector3(1, 1, 1),
      );
      this.surface.visible = true;
      this.sample.visible = false;
    } else {
      this.surface.visible = false;
      this.sample.visible = true;
    }

    if (st.crystalAxes) this.gizmoCry.setDirections(st.crystalAxes);

    // orient the crystal block by U (it already rides the phi group)
    if (st.U) {
      const Um = mat4From3(st.U);
      this.sample.matrix.copy(Um);
      this.sampleEdges.matrix.copy(Um);
    }

    const vis = st.show;
    for (const r of [this.ringMu, this.ringEta, this.ringChi, this.ringPhi])
      r.visible = vis.rings;
    for (const k of Object.keys(this.ringLabels))
      this.ringLabels[k].visible = vis.rings;
    this.floor.visible = vis.floor;
    this.rayHit.visible = vis.rays;
    this.gizmoCry.group.visible = vis.axes;
    this.gizmoCry.title.visible = vis.axes;
    this.gizmoCry.title.visible = vis.axes;

    this.dirty = true;
  }

  /**
   * Stand the whiteboard just outside the detector's own edge in z.
   *
   * `panelWidth` is the panel's extent along eFast, which is the lab z at zero
   * angles, so this is the detector's edge as asked for. It uses the panel's
   * size and not its swung position: the board is bolted to the floor and has
   * no business following delta and gamma around.
   *
   * The floor at boardMinZ keeps it behind the two axis gizmos, which sit at
   * z = ±178. A small detector would otherwise put the board in front of the
   * lab gizmo and hide it.
   */
  _placeBoard(panelWidth) {
    const D = InstrumentScene.DIM;
    const z = Math.max(panelWidth / 2 + D.boardClear, D.boardMinZ);
    this.board.position.z = -z;
  }

  _setLine(line, a, b) {
    const p = line.geometry.attributes.position;
    p.array.set([a[0], a[1], a[2], b[0], b[1], b[2]]);
    p.needsUpdate = true;
    line.geometry.computeBoundingSphere();
  }

  _setSegments(obj, data, colours) {
    if (!data || data.length === 0) {
      obj.visible = false;
      return;
    }
    obj.visible = true;
    const attr = obj.geometry.attributes.position;
    if (attr.array.length < data.length) {
      obj.geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(data.length), 3),
      );
    }
    const a = obj.geometry.attributes.position;
    a.array.set(data);
    a.needsUpdate = true;

    // Per-vertex colour rather than per-vertex alpha: LineBasicMaterial takes
    // its opacity from the material, so brightness is carried in the colour
    // itself, which reads the same way against this dark background.
    if (colours) {
      const existing = obj.geometry.attributes.color;
      if (!existing || existing.array.length < colours.length) {
        obj.geometry.setAttribute(
          "color",
          new THREE.BufferAttribute(new Float32Array(colours.length), 3),
        );
      }
      const c = obj.geometry.attributes.color;
      c.array.set(colours);
      c.needsUpdate = true;
      if (!obj.material.vertexColors) {
        obj.material.vertexColors = true;
        obj.material.needsUpdate = true;
      }
    }
    obj.geometry.setDrawRange(0, data.length / 3);
  }

  /** Push a new detector image (an HTMLCanvasElement) onto the panel. */
  setDetectorImage(canvas) {
    this.detFace.material.map = null;
    this.detTexture.dispose();
    // Mipmaps matter here: without them a minified panel point-samples the
    // image and single-pixel Bragg spots vanish, which reads as "the pattern
    // is not updating" even though the texture is being uploaded every frame.
    this.detTexture = new THREE.CanvasTexture(canvas);
    this.detTexture.minFilter = THREE.LinearMipmapLinearFilter;
    this.detTexture.magFilter = THREE.LinearFilter;
    this.detTexture.generateMipmaps = true;
    this.detTexture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    this.detFace.material.map = this.detTexture;
    this.detFace.material.needsUpdate = true;
    this.dirty = true;
  }

  /** Call when the detector image canvas contents change in place. */
  touchDetectorImage() {
    this.detTexture.needsUpdate = true; // also regenerates the mipmaps
    this.dirty = true;
  }

  /** Keep the whole instrument in shot when the detector distance changes. */
  setSceneScale(distance) {
    const want = Math.max(distance, 150) * 2.6;
    if (want > this.controls.maxDistance) return;
    if (
      this.controls.distance < want * 0.45 ||
      this.controls.distance > want * 3
    ) {
      this.controls.distance = want;
      this.dirty = true;
    }
    this.scale = Math.max(distance, 150);
  }

  resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  render() {
    this.controls.apply(this.camera);

    // Sprite world height that projects to a roughly constant pixel height:
    // px = h * (viewH/2) / (distance * tan(fov/2)).
    const vh = Math.max(this.canvas.clientHeight, 1);
    const fovT = Math.tan((this.camera.fov * Math.PI) / 360);
    const k = (2 * 15 * this.controls.distance * fovT) / vh;
    for (const s of [...Object.values(this.ringLabels), this.detLabel]) {
      if (s) s.scale.set(k * (s.userData.aspect || 2), k, 1);
    }
    // the frame triads carry bigger text than the machine labels
    const gk = k * 1.25;
    for (const s of [
      ...this.gizmoLab.labels,
      ...this.gizmoCry.labels,
      this.gizmoLab.title,
      this.gizmoCry.title,
    ]) {
      if (s) s.scale.set(gk * (s.userData.aspect || 2), gk, 1);
    }

    this.renderer.render(this.scene, this.camera);
    this.dirty = false;
  }
}
