/*! The Game of Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * The pitch–phi control deck.
 *
 * The user-facing half of the second machine: angle rows, the three solve
 * modes (fix alpha / fix phi / explore), the solution list, the batch table,
 * the motor calibration and the zero-angle mounting block. It owns no
 * physics of its own — every number comes from ./pitchphi.js, whose outputs
 * are pinned to the verified standalone calculator by test/parity_pp.mjs.
 *
 * The deck keeps the shared state in one place: st.pp (angles, mount U in the
 * game frame, calibration, solve mode, target). The simulator itself is run
 * by app.js; this file only reads the state, writes DOM, and asks app.js to
 * re-simulate through the requestSim hook.
 *
 * Conventions (see pitchphi.js): angles are internal geometric degrees;
 * the "motor" readouts shown on the solution cards apply the calibration
 * readout = sign * angle + offset. The azimuth sign choice is display-only,
 * exactly as in the standalone tool it replaces.
 */
import * as P from "./physics.js";
import * as PP from "./pitchphi.js";

const $ = (id) => document.getElementById(id);
const DEG = Math.PI / 180;
const f = (x, d = 3) => (isFinite(x) ? x.toFixed(d) : "—");

// motor rows: [key, label, lo, hi, colour]
// pitch runs to +/-180 so a flipped solution (|alpha| > 90, flagged on the
// cards) is reachable rather than silently clamped; the usual working range
// is 0-90.
const ROWS = [
  ["pitch", "pitch", -180, 180, "#f0be46"],
  ["phi", "φ", -180, 180, "#5aa0ff"],
  ["roll", "roll", -180, 180, "#5fc88c"],
  ["tt", "2θ", 0, 180, "#c9d3ea"],
  ["az", "az", -180, 180, "#c9d3ea"],
];

/**
 * Let a number box be typed into without writing over what is in it.
 *
 * Deliberately the same behaviour as the six-circle panel's binder in
 * app.js: half-typed and out-of-range values are ignored while typing, the
 * box is squared up on commit, and a finished out-of-range value is clamped
 * rather than dropped. Kept local so the deck never imports app.js.
 */
function bindTypedNumber(el, lo, hi, apply, settled) {
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    if (!Number.isFinite(v) || v < lo || v > hi) return;
    apply(v);
  });
  el.addEventListener("change", () => {
    const v = parseFloat(el.value);
    if (Number.isFinite(v) && (v < lo || v > hi))
      apply(Math.max(lo, Math.min(hi, v)));
    el.value = settled();
  });
}

/** Fraction- and comma-tolerant number, as the standalone tool parsed them. */
function num(s) {
  s = String(s).trim().replace(",", ".");
  if (s.includes("/")) {
    const [p, q] = s.split("/");
    const v = parseFloat(p) / parseFloat(q);
    if (!isFinite(v)) throw new Error(`bad fraction "${s}"`);
    return v;
  }
  const v = parseFloat(s);
  if (!isFinite(v)) throw new Error(`bad number "${s}"`);
  return v;
}

function triple(s, label) {
  const p = String(s).trim().split(/[\s,;]+/).filter(Boolean);
  if (p.length !== 3) throw new Error(`${label}: need three numbers`);
  return p.map(num);
}

export class PitchPhiDeck {
  /**
   * @param st    the Game state object; this deck reads/writes st.pp and
   *              shares st.B, st.U-less shared fields, st.surfaceHkl
   * @param hooks {requestSim: () => void}
   */
  constructor(st, hooks) {
    this.st = st;
    this.hooks = hooks;
    this.rows = {};
    this.sols = [];
    this._built = false;
    this._keys = { cards: "", sols: "", batch: "", ub: "" };
  }

  // ------------------------------------------------------------- build

  build() {
    if (this._built) return;
    this._built = true;
    const st = this.st;

    // --- angle rows ------------------------------------------------------
    const host = $("ppMotorRows");
    for (const [key, label, lo, hi, colour] of ROWS) {
      const row = document.createElement("div");
      row.className = "motor";
      row.innerHTML =
        `<span class="name" style="color:${colour}">${label}</span>` +
        `<input type="range" min="${lo}" max="${hi}" step="0.01" value="${st.pp.angles[key]}" style="accent-color:${colour}">` +
        `<input type="number" min="${lo}" max="${hi}" step="0.1" value="${st.pp.angles[key]}">`;
      const [range, box] = [row.children[1], row.children[2]];
      const set = (v, keepBox) => {
        v = Math.max(lo, Math.min(hi, v));
        range.value = v;
        if (!keepBox) box.value = Number(v.toFixed(2));
        st.pp.angles[key] = v;
        this.hooks.requestSim();
      };
      range.addEventListener("input", () => set(parseFloat(range.value)));
      bindTypedNumber(
        box,
        lo,
        hi,
        (v) => set(v, true),
        () => Number(st.pp.angles[key].toFixed(2)),
      );
      this.rows[key] = { set, range, box, lo, hi };
      host.appendChild(row);
    }

    // --- target and solve modes ------------------------------------------
    const readTarget = () => [
      +$("ppH").value || 0,
      +$("ppK").value || 0,
      +$("ppL").value || 0,
    ];
    for (const id of ["ppH", "ppK", "ppL"])
      $(id).addEventListener("input", () => {
        st.pp.target = readTarget();
        this.hooks.requestSim();
      });
    document.querySelectorAll("input[name=ppMode]").forEach((r) =>
      r.addEventListener("change", () => {
        st.pp.mode = r.value;
        $("ppModeHint").textContent = {
          alpha: "Find fixes the current pitch, solves φ.",
          phi: "Find fixes the current φ, solves pitch.",
          explore: "Free exploration: the card shows the distance to the Bragg condition.",
        }[r.value];
        this.hooks.requestSim();
      }),
    );
    $("ppFind").addEventListener("click", () => this.find());
    $("ppDrive").addEventListener("click", () => this.drive());
    $("ppAim").addEventListener("click", () => this.aim());

    // --- calibration -----------------------------------------------------
    const bindCal = (id, key) =>
      $(id).addEventListener("change", () => {
        try {
          const v =
            key === "pSign" || key === "fSign" ? +$(id).value : num($(id).value);
          st.pp.cal[key] = v;
          this.hooks.requestSim();
        } catch {
          // a half-typed offset is left alone; the committed value returns
          // on the next valid change
        }
      });
    // seed boxes from state
    $("ppPSign").value = st.pp.cal.pSign;
    $("ppPOff").value = st.pp.cal.pOff;
    $("ppFSign").value = st.pp.cal.fSign;
    $("ppFOff").value = st.pp.cal.fOff;
    $("ppAzSign").value = st.pp.cal.azSign;
    bindCal("ppPSign", "pSign");
    bindCal("ppPOff", "pOff");
    bindCal("ppFSign", "fSign");
    bindCal("ppFOff", "fOff");
    bindCal("ppAzSign", "azSign");

    // --- batch -----------------------------------------------------------
    $("ppBatch").addEventListener("input", () => this.hooks.requestSim());
    $("ppCopy").addEventListener("click", async () => {
      const rows = [...$("ppTable").querySelectorAll("tr")].map((tr) =>
        [...tr.children].map((td) => td.textContent).join("\t"),
      );
      let label = "Copied";
      try {
        await navigator.clipboard.writeText(rows.join("\n"));
      } catch {
        label = "Copy refused: select the table";
      }
      $("ppCopy").textContent = label;
      setTimeout(() => {
        $("ppCopy").textContent = "Copy table as TSV";
      }, 1600);
    });

    // --- mounting --------------------------------------------------------
    for (const id of ["ppSH", "ppSK", "ppSL"])
      $(id).addEventListener("input", () => {
        const s = [+$("ppSH").value || 0, +$("ppSK").value || 0, +$("ppSL").value || 0];
        st.surfaceHkl = s;
        // keep the six-circle section's copies of the same state in step
        $("sh").value = s[0];
        $("sk").value = s[1];
        $("sl").value = s[2];
        this.hooks.requestSim();
      });
    $("ppMount").addEventListener("click", () => this.mount());
    $("ppUBConv").addEventListener("change", () => (this._keys.ub = ""));
    $("ppUBCopy").addEventListener("click", async () => {
      let label = "Copied";
      try {
        await navigator.clipboard.writeText($("ppUB").textContent);
      } catch {
        label = "Copy refused: select the text";
        const sel = window.getSelection();
        if (sel) sel.selectAllChildren($("ppUB"));
      }
      $("ppUBCopy").textContent = label;
      setTimeout(() => {
        $("ppUBCopy").textContent = "Copy";
      }, 1600);
    });

    // seed the solve-mode hint from the saved mode
    $("ppModeHint").textContent = {
      alpha: "Find fixes the current pitch, solves φ.",
      phi: "Find fixes the current φ, solves pitch.",
      explore: "Free exploration: the card shows the distance to the Bragg condition.",
    }[st.pp.mode];
  }

  /** Called on every instrument switch; clears caches so sync() redraws. */
  activate(ppActive) {
    this._keys = { cards: "", sols: "", batch: "", ub: "" };
    this._found = false;
    if (ppActive && !this._built) this.build();
  }

  // ------------------------------------------------------------- helpers

  /** Q of the target at zero angles, in the instrument's frame (2pi/Angstrom). */
  quOf(hkl = this.st.pp.target) {
    const { st } = this;
    return P.matVec(
      P.matMul(PP.GAME_TO_PP, P.matMul(st.pp.U, st.B)),
      hkl,
    );
  }

  /** A cheap cache key for anything derived from Q at zero angles. */
  _quKey(QU) {
    return `${QU[0].toFixed(12)},${QU[1].toFixed(12)},${QU[2].toFixed(12)}`;
  }

  motorP(a) {
    const c = this.st.pp.cal;
    return c.pSign * a + c.pOff;
  }

  motorF(p) {
    const c = this.st.pp.cal;
    return c.fSign * p + c.fOff;
  }

  statusText(e) {
    return e.ok
      ? "both beams above surface"
      : e.flipped
        ? "sample flipped (|α| > 90°)"
        : e.alpha <= 0
          ? "incident beam below surface"
          : "exit beam below surface";
  }

  // ------------------------------------------------------------- solver UI

  find() {
    this._found = true;
    const { st } = this;
    const msg = $("ppMsg");
    msg.style.color = "var(--bad)";
    try {
      const hkl = st.pp.target;
      if (!hkl.some(Boolean)) {
        msg.textContent = "0 0 0 is the direct beam; give a reflection.";
        return;
      }
      const k = (2 * Math.PI) / st.wavelength;
      const QU = this.quOf(hkl);
      const a = st.pp.angles;
      this.sols = PP.solutions(QU, st.pp.mode, a.pitch, a.phi, a.roll, k, st.pp.cal.azSign);
      st.pp.sols = this.sols;
      const sel = $("ppSols");
      sel.innerHTML = "";
      const q = P.norm(QU);
      if (st.pp.mode === "explore") {
        const e = this.sols[0];
        const o = document.createElement("option");
        o.textContent =
          `current setting — ε = ${e.eps.toExponential(2)} Å⁻¹` +
          (Math.abs(e.eps) < 1e-3 ? " (on Bragg)" : "");
        sel.appendChild(o);
        msg.style.color = Math.abs(e.eps) < 1e-3 ? "var(--good)" : "var(--bad)";
        msg.textContent =
          `|Q| = ${q.toFixed(4)} Å⁻¹, d = ${((2 * Math.PI) / q).toFixed(4)} Å, ` +
          `2θ_B = ${f((2 * Math.asin(Math.min(q / (2 * k), 1)) * 180) / Math.PI, 3)}°`;
        return;
      }
      if (!this.sols.length) {
        const o = document.createElement("option");
        o.textContent = "no solution at this fixed angle";
        sel.appendChild(o);
        msg.textContent =
          st.pp.mode === "alpha"
            ? `no φ works at pitch = ${f(a.pitch, 2)}°: this reflection is reachable only for α in the band on the card`
            : `no α puts this reflection on the Ewald sphere at φ = ${f(a.phi, 2)}°`;
        return;
      }
      let blocked = 0;
      this.sols.forEach((e, i) => {
        const o = document.createElement("option");
        o.value = i;
        o.textContent =
          `pitch motor ${f(this.motorP(e.alpha))}°  ·  φ motor ${f(this.motorF(e.phi))}°  →  ` +
          `2θ ${f(e.tth)}°  az ${f(e.az)}°` +
          (e.anyPhi ? "  (specular: any φ works)" : "") +
          (e.ok ? "" : `  — ${this.statusText(e)}`);
        if (!e.ok) blocked++;
        sel.appendChild(o);
      });
      const first = [...sel.options].find((o) => this.sols[+o.value] && this.sols[+o.value].ok);
      (first || sel.options[0]).selected = true;
      msg.style.color = "var(--good)";
      msg.textContent =
        `|Q| = ${q.toFixed(4)} Å⁻¹, d = ${((2 * Math.PI) / q).toFixed(4)} Å, ` +
        `${this.sols.length} solution(s)` +
        (blocked ? `, ${blocked} with a beam below the surface.` : ".");
    } catch (err) {
      msg.textContent = err.message;
    }
  }

  drive() {
    const { st } = this;
    const sel = $("ppSols");
    const i = +sel.value;
    const e = this.sols[i];
    if (!e) return;
    // full-precision geometric values, not the rounded motor readouts
    this.rows.pitch.set(e.alpha, true);
    this.rows.phi.set(e.phi, true);
    this.rows.tt.set(e.tth, true);
    this.rows.az.set(PP.wrap(e.az), true);
    this.hooks.requestSim();
  }

  aim() {
    const { st } = this;
    const msg = $("ppMsg");
    try {
      const hkl = st.pp.target;
      if (!hkl.some(Boolean)) {
        msg.style.color = "var(--bad)";
        msg.textContent = "0 0 0 is the direct beam; give a reflection.";
        return;
      }
      const k = (2 * Math.PI) / st.wavelength;
      const a = st.pp.angles;
      const e = PP.evaluate(this.quOf(hkl), a.pitch, a.phi, a.roll, k, st.pp.cal.azSign);
      const { tth, az } = PP.kfToAngles(e.kf, st.pp.cal.azSign);
      this.rows.tt.set(tth, true);
      this.rows.az.set(PP.wrap(az), true);
      const onSphere = Math.abs(e.eps) <= 1e-3;
      msg.style.color = onSphere ? "var(--good)" : "var(--bad)";
      msg.textContent =
        `2θ to ${tth.toFixed(2)}°, az to ${az.toFixed(2)}°` +
        (onSphere
          ? " — the reflection is on the Ewald sphere, so it will be on the panel."
          : " — but it is not on the Ewald sphere at these angles, so nothing will be there: use Find first.");
      this.hooks.requestSim();
    } catch (err) {
      msg.style.color = "var(--bad)";
      msg.textContent = err.message;
    }
  }

  mount() {
    const { st } = this;
    const msg = $("ppMountMsg");
    msg.style.color = "var(--bad)";
    try {
      if (!st.pp.init) {
        msg.textContent = "Switch to the pitch–phi instrument first.";
        return;
      }
      const s = [+$("ppSH").value || 0, +$("ppSK").value || 0, +$("ppSL").value || 0];
      if (!s.some(Boolean)) {
        msg.textContent = "Give a non-zero surface (hkl) first.";
        return;
      }
      const r = [+$("ppRH").value || 0, +$("ppRK").value || 0, +$("ppRL").value || 0];
      if (!r.some(Boolean)) {
        msg.textContent = "Give a non-zero in-plane reference first.";
        return;
      }
      const kind = $("ppRKind").value;
      const nC = P.unit(P.crystalVector(st.B, s, "hkl"));
      const ref = P.crystalVector(st.B, r, kind);
      const Upp = PP.mountOrientation(nC, ref, $("ppRDir").value);
      st.pp.U = P.matMul(PP.PP_TO_GAME, Upp);
      st.surfaceHkl = s;
      $("sh").value = s[0];
      $("sk").value = s[1];
      $("sl").value = s[2];
      msg.style.color = "var(--good)";
      msg.textContent =
        `Surface normal (${s.join(" ")}) mounted vertical at pitch = φ = 0; ` +
        `${kind === "hkl" ? "(hkl)" : "[uvw]"} ${r.join(" ")} points along ` +
        `${$("ppRDir").selectedOptions[0].textContent}.`;
      this.hooks.requestSim();
    } catch (err) {
      msg.textContent = err.message;
    }
  }

  // ------------------------------------------------------------- sync

  /** Called from simulate() every frame while this instrument is active. */
  sync() {
    const { st } = this;
    if (!st.pp.init) return;
    const k = (2 * Math.PI) / st.wavelength;
    const QU = this.quOf();
    const a = st.pp.angles;

    // cards
    const key = `${this._quKey(QU)}|${st.wavelength}|${a.roll}`;
    if (key !== this._keys.cards) {
      this._keys.cards = key;
      const q = P.norm(QU);
      const d = (2 * Math.PI) / q;
      const sinth = q / (2 * k);
      const thB = sinth <= 1 ? (Math.asin(sinth) * 180) / Math.PI : NaN;
      const spec = PP.isSpecular(QU);
      const band = isFinite(thB) ? PP.alphaBand(QU, k, a.roll) : [];
      const bandTxt = spec
        ? isFinite(thB)
          ? `${f(thB, 2)} only`
          : "—"
        : band.length
          ? band.map((r) => `${f(r[0], 1)}–${f(r[1], 1)}`).join(", ")
          : "none";
      const card = (k2, v, u) =>
        `<div class="stat"><span class="k">${k2}</span><span class="v">${v}</span> <span class="u">${u}</span></div>`;
      $("ppCards").innerHTML =
        card("|Q|", f(q, 4), "Å⁻¹") +
        card("d", f(d, 4), "Å") +
        card("2θ_B", isFinite(thB) ? f(2 * thB, 3) : "unreachable", "°") +
        card("Q ⟂ surface", f(QU[1], 4), "Å⁻¹") +
        card("Q ∥ surface", f(Math.hypot(QU[0], QU[2]), 4), "Å⁻¹") +
        card("α reachable", bandTxt, "°");
    }

    // solution list stays live once Find has been pressed: from then on the
    // list follows the fixed angle and the calibration silently; before the
    // first press nothing appears, so the panel is not pre-empting the user
    const solKey = `${this._quKey(QU)}|${st.pp.mode}|${a.pitch}|${a.phi}|${a.roll}|${st.wavelength}|${st.pp.cal.pSign}|${st.pp.cal.pOff}|${st.pp.cal.fSign}|${st.pp.cal.fOff}|${st.pp.cal.azSign}`;
    if (this._found && solKey !== this._keys.sols) this.find();
    this._keys.sols = solKey;

    // batch
    const batchKey = `${$("ppBatch").value}|${this._quKey(QU)}|${st.pp.mode}|${a.pitch}|${a.phi}|${a.roll}|${st.wavelength}|${st.pp.cal.azSign}`;
    if (batchKey !== this._keys.batch) {
      this._keys.batch = batchKey;
      this.renderBatch(k, QU);
    }

    // UB in the native frame at zero angles
    const conv = $("ppUBConv").value;
    const ubRaw = PP.zeroAngleUB(st.pp.U, st.B);
    const factor = conv === "1d" ? 1 / (2 * Math.PI) : conv === "lambda" ? st.wavelength / (2 * Math.PI) : 1;
    const ubTxt = ubRaw
      .map((row) =>
        row
          .map((x) => {
            const v = x * factor;
            const r = Math.abs(v) < 5e-7 ? 0 : v;
            return (r >= 0 ? "+" : "") + r.toFixed(6);
          })
          .join("  "),
      )
      .join("\n");
    if (ubTxt !== this._keys.ub) {
      this._keys.ub = ubTxt;
      $("ppUB").textContent = ubTxt;
    }
  }

  renderBatch(k, QU) {
    const { st } = this;
    const lines = $("ppBatch").value
      .split(/\n/)
      .map((s) => s.trim())
      .filter(Boolean);
    const rows = [];
    for (const ln of lines) {
      let hkl;
      try {
        hkl = triple(ln, "batch");
      } catch {
        rows.push({ hkl: ln, bad: "parse" });
        continue;
      }
      const Q = this.quOf(hkl);
      const q = P.norm(Q);
      const d = (2 * Math.PI) / q;
      if (q / (2 * k) > 1) {
        rows.push({ hkl: ln, d, bad: "|Q|>2k" });
        continue;
      }
      const s = PP.solutions(Q, st.pp.mode, st.pp.angles.pitch, st.pp.angles.phi, st.pp.angles.roll, k, st.pp.cal.azSign);
      if (st.pp.mode === "explore") {
        rows.push({ hkl: ln, d, e: s[0], eps: s[0].eps });
        continue;
      }
      if (!s.length) {
        rows.push({ hkl: ln, d, tth: (2 * Math.asin(q / (2 * k)) * 180) / Math.PI, bad: "no solution" });
        continue;
      }
      for (const e of s) rows.push({ hkl: ln, d, e });
    }
    const H = [
      "h k l",
      "d (Å)",
      "2θ (°)",
      "pitch motor",
      "φ motor",
      "α (°)",
      "β (°)",
      "az (°)",
      st.pp.mode === "explore" ? "ε (Å⁻¹)" : "status",
    ];
    let html = "<thead><tr>" + H.map((h) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>";
    for (const r of rows) {
      if (r.bad) {
        html +=
          `<tr class="bad"><td>${r.hkl}</td><td>${f(r.d)}</td><td>${f(r.tth)}</td>` +
          `<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td class="st">${r.bad}</td></tr>`;
        continue;
      }
      const e = r.e;
      html +=
        `<tr class="${e.ok ? "good" : "bad"}"><td>${r.hkl}</td><td>${f(r.d)}</td><td>${f(e.tth)}</td>` +
        `<td>${f(this.motorP(e.alpha))}</td><td>${f(this.motorF(e.phi))}</td>` +
        `<td>${f(e.alpha)}</td><td>${f(e.beta)}</td><td>${f(e.az)}</td>` +
        `<td class="st">${r.eps !== undefined ? "ε=" + r.eps.toFixed(3) : e.ok ? "ok" : this.statusText(e)}</td></tr>`;
    }
    $("ppTable").innerHTML = html + "</tbody>";
  }
}
