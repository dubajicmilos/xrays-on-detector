/*! The Game of Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * The pitch–phi diffractometer in the shared three.js scene.
 *
 * A separate rig rather than something the six-circle hierarchy morphs into:
 * the two machines share the renderer, camera, detector panel, rays, floor
 * and whiteboard, but their circles are physically different hardware, so
 * each gets its own drawing. The rig is built once, hidden until its
 * instrument is selected, and then only group matrices are touched per frame
 * (roll carries pitch carries phi, exactly as the real stack does).
 *
 * Axes in the site's lab frame (+x up, +y beam, +z horizontal), from
 * Z_game = R_y(roll) . R_z(-alpha) . R_x(phi):
 *   roll  about +y (the beam axis), outermost
 *   pitch about +z, the horizontal axis normal to the beam
 *   phi   about +x, the sample normal at zero pitch
 *
 * The detector itself is the shared panel driven by the active Detector
 * frame; the rig draws the azimuth arc that shows where on the cone the
 * arm stands (from the vertical, sweeping by the azimuth, at the detector's
 * own distance so the arc ends where the panel hangs) and the crystal
 * outline, so pitch, phi and roll visibly turn the sample.
 */
import * as THREE from "../lib/three.module.js";

import { InstrumentScene, mat4From3, makeLabel } from "./scene.js";
import { rotX, rotY, rotZ } from "./physics.js";
import { detectorDir, ppToGame } from "./pitchphi.js";

const DEG = Math.PI / 180;

const PPCOL = {
  pitch: 0xf0be46,
  phi: 0x5aa0ff,
  roll: 0x5fc88c,
  arc: 0xe0c060,
};

export class PitchPhiRig {
  /** @param parent the THREE.Scene the shared instrument lives in. */
  constructor(parent) {
    const D = InstrumentScene.DIM;
    this.root = new THREE.Group();
    this.root.visible = false;
    parent.add(this.root);

    this.gRoll = new THREE.Group();
    this.gPitch = new THREE.Group();
    this.gPhi = new THREE.Group();
    for (const g of [this.gRoll, this.gPitch, this.gPhi]) g.matrixAutoUpdate = false;
    this.root.add(this.gRoll);
    this.gRoll.add(this.gPitch);
    this.gPitch.add(this.gPhi);

    const ring = (radius, colour, axis) => {
      const geo = new THREE.TorusGeometry(radius, D.tube, 12, 96);
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

    // each ring hangs on the group of the axis it belongs to, like the
    // six-circle: the ring for a circle rides the circle it rotates in
    this.ringRoll = ring(D.ringEta, PPCOL.roll, "y");
    this.ringPitch = ring(D.ringChi, PPCOL.pitch, "z");
    this.ringPhi = ring(D.ringPhi, PPCOL.phi, "x");
    this.root.add(this.ringRoll);
    this.gRoll.add(this.ringPitch);
    this.gPitch.add(this.ringPhi);

    // labels float at their ring's rim; the scene scales sprites per frame
    this._labels = [];
    const label = (text, colour, pos) => {
      const s = makeLabel(text, colour, 40);
      s.position.set(pos[0], pos[1], pos[2]);
      this.root.add(s);
      this._labels.push(s);
      return s;
    };
    // each at its own point on its own rim, so the three do not stack above
    // the sample: roll's ring stands across the beam (x-z), pitch's along it
    // (x-y), and phi's lies flat (y-z)
    const q = 1.16 * Math.SQRT1_2;
    label("roll", "#5fc88c", [D.ringEta * q, 0, D.ringEta * q]);
    label("pitch", "#f0be46", [D.ringChi * q, D.ringChi * q, 0]);
    label("phi", "#5aa0ff", [0, -D.ringPhi * q, D.ringPhi * q]);

    // The crystal itself, as the same outline block the six-circle draws.
    // The rings are rotationally symmetric, so without this nothing on the
    // machine moves when pitch or phi is dialled and the sample reads as
    // welded in place; the outline rides the phi group (world = Z . U) and
    // turns with every sample axis. The surface slab, shared with the
    // six-circle, keeps showing the datum plane.
    this.sampleEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(
        new THREE.BoxGeometry(D.sample, D.sample * 0.72, D.sample * 0.55),
      ),
      new THREE.LineBasicMaterial({ color: 0x7f8db0 }),
    );
    this.sampleEdges.matrixAutoUpdate = false;
    this.gPhi.add(this.sampleEdges);

    // detector azimuth arc: from the vertical (azimuth 0) to the arm
    this._arcN = 96;
    const arcGeo = new THREE.BufferGeometry();
    arcGeo.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array((this._arcN + 1) * 3), 3),
    );
    this.arc = new THREE.Line(
      arcGeo,
      new THREE.LineBasicMaterial({
        color: PPCOL.arc,
        transparent: true,
        opacity: 0.85,
      }),
    );
    this.arc.frustumCulled = false;
    this.root.add(this.arc);
  }

  setVisible(v) {
    this.root.visible = v;
  }

  /** Sprites the scene should keep pixel-sized (scene.render does the scaling). */
  labels() {
    return this._labels;
  }

  /**
   * @param pose {angles: {pitch, phi, roll, tt, az}, azSign, centre, U, rings}
   *   centre: the detector panel centre in mm, from the active Detector frame
   *   U: the zero-angle mount (game frame); the crystal outline rides gPhi,
   *      so its world matrix is Z . U exactly
   */
  update(pose) {
    const { angles, azSign, centre } = pose;
    this.gRoll.matrix.copy(mat4From3(rotY(angles.roll * DEG)));
    this.gPitch.matrix.copy(mat4From3(rotZ(-angles.pitch * DEG)));
    this.gPhi.matrix.copy(mat4From3(rotX(angles.phi * DEG)));
    if (pose.U) this.sampleEdges.matrix.copy(mat4From3(pose.U));

    // the Circles toggle reaches this rig the same way it reaches the
    // six-circle's rings
    const vis = pose.rings !== false;
    this.ringRoll.visible = vis;
    this.ringPitch.visible = vis;
    this.ringPhi.visible = vis;
    for (const s of this._labels) s.visible = vis;

    const R = Math.hypot(centre[0], centre[1], centre[2]);
    const p = this.arc.geometry.attributes.position;
    for (let i = 0; i <= this._arcN; i++) {
      const a = angles.az * (i / this._arcN);
      const kh = ppToGame(detectorDir(angles.tt, a, azSign));
      p.array[3 * i] = kh[0] * R;
      p.array[3 * i + 1] = kh[1] * R;
      p.array[3 * i + 2] = kh[2] * R;
    }
    p.needsUpdate = true;
    this.arc.geometry.computeBoundingSphere();
    this.arc.visible = Math.abs(angles.az) > 0.01 && R > 0;
  }
}
