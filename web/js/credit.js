/*! The Game of Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Authorship mark: one place owns the credit.
 *
 * The sidebar line, the signature on the detector panel in the 3D view, the
 * console banner and the document metadata all read the strings below, so the
 * copies of the attribution cannot drift apart.
 *
 * This is licence support, not DRM. Anyone holding the source can delete it,
 * and nothing here pretends otherwise. What it buys: an honest copy carries
 * the credit without anyone having to remember it, a screenshot of the
 * instrument carries it too, and removing it takes an edit to the source
 * rather than a lapse of attention. The MIT terms already require this notice
 * to survive in "all copies or substantial portions of the Software", so a
 * stripped copy is a breach of the licence, not a grey area.
 */

export const CREDIT = Object.freeze({
  title: "The Game of Diffraction",
  author: "Miloš Dubajić",
  year: 2026,
  home: "https://dubajicmilos.github.io/diffraction/",
  source: "https://github.com/dubajicmilos/xrays-on-detector",
  licence: "MIT",
});

// ------------------------------------------------------- whiteboard panel

const MARK = `© ${CREDIT.year} ${CREDIT.author}`;
const FACE = '"Segoe UI", system-ui, -apple-system, sans-serif';
// Falls back through what Windows, macOS and Linux each actually ship. The
// layout is left-aligned throughout, so a substitute with different letter
// widths shifts the line endings and nothing else.
const HAND = '"Segoe Script", "Bradley Hand", "Comic Sans MS", cursive';
const POSTER_FACE = 'Impact, "Arial Black", "Haettenschweiler", sans-serif';

const PX_PER_MM = 3.5; // keeps a 560 x 370 board just under a 2048 px texture
const FRAME = 16; // mm of aluminium surround

/** Written on the board. The physics the simulator actually runs on. */
const EQUATIONS = [
  "n λ = 2 d sin θ",
  "|Q| = 4π sin θ / λ",
  "Q = h a* + k b* + l c*",
  "|k| = 2π / λ",
];

/**
 * The two posters, pinned to the right of the writing. Widths are in mm on the
 * board; each image keeps its own aspect ratio, so one comes out landscape and
 * the other portrait, as the source photographs are.
 */
const POSTERS = [
  {
    file: "braggs.jpg",
    x: 240,
    y: 24,
    w: 82,
    tilt: -2.5,
    name: "W. H. & W. L. BRAGG",
  },
  { file: "laue.jpg", x: 328, y: 30, w: 70, tilt: 3, name: "M. VON LAUE" },
];

// Everything printed sits inside texture x 25..400 and y 20..200. That is not
// arbitrary: at the opening camera the detector hides the board below x = 0,
// the pane's right edge cuts it at x = 400, and the goniometer is read through
// anything below y = 200. Measured, not guessed; re-measure before moving the
// board or changing its size.

/**
 * Draw the whiteboard that stands alongside the beam, as a canvas to be used
 * as a texture.
 *
 * The credit lives here rather than on the detector image, where it covered
 * Bragg spots. Millimetres are the unit throughout.
 *
 * The photographs load over the network, so the board is drawn once without
 * them and again when they arrive; `onReady` is how the scene knows to push
 * the texture up a second time.
 */
export function whiteboardTexture(lengthMm, heightMm, onReady) {
  const c = document.createElement("canvas");
  c.width = Math.round(lengthMm * PX_PER_MM);
  c.height = Math.round(heightMm * PX_PER_MM);

  const loaded = new Map();
  const paint = () => drawBoard(c, lengthMm, heightMm, loaded);
  paint();

  let pending = POSTERS.length;
  const settle = () => {
    if (--pending > 0) return;
    paint();
    onReady?.();
  };
  for (const p of POSTERS) {
    const im = new Image();
    im.onload = () => {
      loaded.set(p.file, im);
      settle();
    };
    im.onerror = () => {
      // Not fatal: the poster keeps its frame and heading, and the credit is
      // untouched. Say so rather than leaving an empty rectangle unexplained.
      console.warn(`whiteboard: could not load img/${p.file}`);
      settle();
    };
    im.src = new URL(`../img/${p.file}`, import.meta.url).href;
  }
  return c;
}

function drawBoard(c, lengthMm, heightMm, loaded) {
  const px = (v) => v * PX_PER_MM;
  const g = c.getContext("2d");
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, c.width, c.height);

  // -- aluminium surround and pen tray ------------------------------------
  const rail = g.createLinearGradient(0, 0, 0, c.height);
  rail.addColorStop(0, "#9aa3b4");
  rail.addColorStop(0.5, "#6d7688");
  rail.addColorStop(1, "#565e6e");
  g.fillStyle = rail;
  g.fillRect(0, 0, c.width, c.height);

  // -- writing surface ----------------------------------------------------
  // Not pure white. A white rectangle this size glares in a scene lit for a
  // dark hutch and pulls the eye off the diffraction pattern.
  const face = g.createLinearGradient(0, px(FRAME), px(lengthMm), px(heightMm));
  face.addColorStop(0, "#e3e7ee");
  face.addColorStop(0.55, "#d6dbe4");
  face.addColorStop(1, "#c7cdd8");
  g.fillStyle = face;
  g.fillRect(
    px(FRAME),
    px(FRAME),
    px(lengthMm - 2 * FRAME),
    px(heightMm - 2 * FRAME),
  );

  // a wiped-off sheen, so the surface reads as glossy rather than as paper
  g.save();
  g.beginPath();
  g.rect(
    px(FRAME),
    px(FRAME),
    px(lengthMm - 2 * FRAME),
    px(heightMm - 2 * FRAME),
  );
  g.clip();
  g.globalAlpha = 0.5;
  const sheen = g.createLinearGradient(
    px(60),
    px(FRAME),
    px(260),
    px(heightMm),
  );
  sheen.addColorStop(0, "rgba(255,255,255,0)");
  sheen.addColorStop(0.5, "rgba(255,255,255,0.5)");
  sheen.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = sheen;
  g.fillRect(0, 0, c.width, c.height);
  g.restore();

  // pen tray along the bottom rail, with two markers and an eraser on it
  const trayY = heightMm - FRAME + 3;
  g.fillStyle = "#7d8698";
  g.fillRect(px(28), px(trayY), px(lengthMm - 56), px(6));
  g.fillStyle = "#3c4657";
  g.fillRect(px(28), px(trayY + 5), px(lengthMm - 56), px(1.6));
  const pens = [
    [56, "#2f6fd0"],
    [86, "#c0392b"],
  ];
  for (const [x, colour] of pens) {
    g.fillStyle = colour;
    g.fillRect(px(x), px(trayY - 2.6), px(26), px(4.4));
  }
  g.fillStyle = "#2b3242";
  g.fillRect(px(126), px(trayY - 3.4), px(30), px(5.6));

  // -- writing -------------------------------------------------------------
  const left = 40;
  g.textAlign = "left";
  g.textBaseline = "alphabetic";

  g.fillStyle = "#1b2a52";
  g.font = `${px(14)}px ${HAND}`;
  g.fillText(CREDIT.title, px(left), px(46));

  g.fillStyle = "#3d4a68";
  g.font = `${px(13)}px ${HAND}`;
  g.fillText(MARK, px(left), px(68));

  g.strokeStyle = "#8b96ad";
  g.lineWidth = px(0.9);
  g.beginPath();
  g.moveTo(px(left), px(78));
  g.lineTo(px(left + 190), px(78));
  g.stroke();

  // Marker colours, alternating the way a board actually gets written on.
  const ink = ["#17224a", "#a8341f", "#17224a", "#1c6444"];
  g.font = `${px(18)}px ${HAND}`;
  EQUATIONS.forEach((eq, i) => {
    g.fillStyle = ink[i % ink.length];
    g.fillText(eq, px(left), px(106 + i * 27));
  });

  // Half-wiped marker low down, where the board is read through the circles.
  // Keeps the empty lower half from looking like a blank rectangle.
  g.save();
  g.globalAlpha = 0.07;
  g.strokeStyle = "#3d4a68";
  g.lineCap = "round";
  for (const [sx, sy, w, lw] of [
    [70, 248, 96, 11],
    [104, 272, 145, 15],
    [58, 298, 74, 9],
  ]) {
    g.lineWidth = px(lw);
    g.beginPath();
    g.moveTo(px(sx), px(sy));
    g.lineTo(px(sx + w), px(sy - 4));
    g.stroke();
  }
  g.restore();

  // -- posters -------------------------------------------------------------
  for (const p of POSTERS) drawPoster(g, px, p, loaded.get(p.file));
}

/** One wanted poster, tilted as though stuck on with magnets. */
function drawPoster(g, px, p, img) {
  const pad = 6;
  const inner = p.w - 2 * pad;
  // Height follows the photograph, so a landscape and a portrait source give
  // two differently proportioned posters rather than one stretched to fit.
  const shot = img ? (inner * img.height) / img.width : inner * 0.75;
  const head = p.w * 0.2;
  const h = pad + head + 5 + shot + 4 + p.w * 0.09 + pad;

  g.save();
  g.translate(px(p.x + p.w / 2), px(p.y + h / 2));
  g.rotate((p.tilt * Math.PI) / 180);
  g.translate(px(-p.w / 2), px(-h / 2));

  g.shadowColor = "rgba(20, 26, 40, 0.45)";
  g.shadowBlur = px(4);
  g.shadowOffsetY = px(2.5);
  g.fillStyle = "#f2ead6"; // aged paper
  g.fillRect(0, 0, px(p.w), px(h));
  g.shadowColor = "transparent";
  g.shadowBlur = 0;
  g.shadowOffsetY = 0;

  g.strokeStyle = "#4a3f2c";
  g.lineWidth = px(1.1);
  g.strokeRect(px(2.5), px(2.5), px(p.w - 5), px(h - 5));

  g.textAlign = "center";
  g.fillStyle = "#2b2114";
  g.font = `${px(head)}px ${POSTER_FACE}`;
  g.fillText("WANTED", px(p.w / 2), px(pad + head * 0.86));

  const sx = pad;
  const sy = pad + head + 5;
  if (img) {
    g.drawImage(img, px(sx), px(sy), px(inner), px(shot));
  } else {
    g.fillStyle = "#d8cdb2";
    g.fillRect(px(sx), px(sy), px(inner), px(shot));
  }
  g.strokeStyle = "#4a3f2c";
  g.lineWidth = px(0.8);
  g.strokeRect(px(sx), px(sy), px(inner), px(shot));

  g.fillStyle = "#3a2f1f";
  g.font = `${px(p.w * 0.075)}px ${POSTER_FACE}`;
  g.fillText(p.name, px(p.w / 2), px(sy + shot + p.w * 0.085));

  // magnets
  for (const mx of [p.w * 0.18, p.w * 0.82]) {
    g.fillStyle = "#c0392b";
    g.beginPath();
    g.arc(px(mx), px(4.5), px(3.2), 0, Math.PI * 2);
    g.fill();
    g.fillStyle = "rgba(255,255,255,0.35)";
    g.beginPath();
    g.arc(px(mx - 0.9), px(3.6), px(1.1), 0, Math.PI * 2);
    g.fill();
  }
  g.restore();
}
// --------------------------------------------------------------- sidebar

const LINE_ID = "credit";

/**
 * The sibling apps in this repo share an author, a licence and a source, and
 * differ only in what the page is called. `work` overrides those two fields;
 * everything the licence actually turns on is fixed.
 */
const workOf = (work) => ({
  title: work?.title || CREDIT.title,
  home: work?.home || CREDIT.home,
});

function buildLine(work) {
  const { title, home } = workOf(work);
  const el = document.createElement("footer");
  el.className = "tiny";
  el.id = LINE_ID;
  const link = (href, text) =>
    `<a href="${href}" target="_blank" rel="noopener">${text}</a>`;
  el.innerHTML =
    `${link(home, title)} · © ${CREDIT.year} ${CREDIT.author}` +
    `<br />${CREDIT.licence} licensed · ${link(CREDIT.source, "source")} · ` +
    `keep this notice in copies`;
  return el;
}

/**
 * Put the credit at the foot of the control panel and keep it there.
 *
 * The two observers restore the line if its element is removed or its text is
 * emptied. That does not make the mark unremovable and is not meant to: it
 * makes removing it a change to this file, which is the difference between an
 * accident and a decision. They are scoped to the panel's direct children and
 * to the line itself, not to the whole subtree, so the readouts that rewrite
 * their text every frame do not wake them.
 */
export function mountCredit(host, work = null) {
  let node = buildLine(work);
  host.appendChild(node);

  const intact = () =>
    node.isConnected && node.textContent.includes(CREDIT.author);
  const restore = () => {
    if (intact()) return;
    // Clear the tampered line out first. Appending alongside it would leave
    // two elements sharing the id, with the emptied or rewritten one ahead of
    // the real one in document order, which is worse than not restoring.
    for (const stale of host.querySelectorAll(`#${LINE_ID}`)) stale.remove();
    node = buildLine(work);
    host.appendChild(node);
    watchText();
  };

  let textWatcher = null;
  function watchText() {
    textWatcher?.disconnect();
    textWatcher = new MutationObserver(restore);
    textWatcher.observe(node, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }
  watchText();
  new MutationObserver(restore).observe(host, { childList: true });
}

/** Name the author in the console, where anyone inspecting a copy will look. */
export function logCredit(work = null) {
  const { title, home } = workOf(work);
  console.log(
    `%c${title}%c\n© ${CREDIT.year} ${CREDIT.author} · ${home}` +
      `\nSource: ${CREDIT.source} (${CREDIT.licence}).` +
      `\nThe licence asks that this notice travel with any copy.`,
    "font-weight:600;font-size:13px",
    "font-weight:400",
  );
}
