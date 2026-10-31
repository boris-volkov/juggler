// Juggling simulation.
//
// World units are metres and seconds, y points up, gravity is real (9.81).
// Each hand moves around an ellipse. The ball path is built so that it is
// C1-continuous everywhere: every flight parabola leaves one hand ellipse
// tangentially (same direction AND speed as the hand) and arrives at the
// next hand ellipse tangentially, then rides the bottom of that ellipse
// until the next throw. In a cascade the two parabolas plus the two dwell
// arcs form the figure-8; the ellipses touch it along its two low points.
//
// The solver works in the "standard" sense (right hand turns clockwise,
// throws inside, catches outside). The time-reverse of a gravity parabola
// is still a gravity parabola, so the reverse cascade (throw outside,
// catch inside) is exactly the standard pattern evaluated at -t.

(function () {
  'use strict';

  const G = 9.81;
  const BALL_R = 0.024;
  const MIN_DWELL = 0.08;    // beats
  const BULGE = 0.3;         // outer-bottom bulge of the hand loops
  const TAU = Math.PI * 2;

  const mod = (x, m) => ((x % m) + m) % m;

  // 1 and 2 balls reuse the 3-ball rhythm with gaps so the figure-8 survives.
  function patternFor(n) {
    if (n === 1) return [3, 0, 0];
    if (n === 2) return [3, 3, 0];
    return [n];
  }

  // ---------- geometry ----------

  // Hand loop: an ellipse with one bottom quarter pushed out and down, like
  // a racing line swinging wide through a U-turn. The bulge sits on the
  // throwing side (q = +1 outside, -1 inside), just before the throw, so the
  // hand has a longer run to accelerate along the throw. The push is
  //   w = sin^2(th) * (1 + q cos th) / 2   (lower half only)
  // which peaks below the centre on the throwing side, fades toward the
  // other side, and is zero with zero slope at the centre line, so the sides
  // keep their vertical tangents and the joint with the round upper half is
  // smooth.
  function makeGeom(X, a, b, e, q) {
    const w = (th) => {
      const s = Math.sin(th);
      return s < 0 ? s * s * (1 + q * Math.cos(th)) / 2 : 0;
    };
    const dw = (th) => {
      const s = Math.sin(th), c = Math.cos(th);
      return s < 0 ? s * c * (1 + q * c) - q * s * s * s / 2 : 0;
    };
    const dP = (th) => ({
      x: a * (-Math.sin(th) + q * e * dw(th)),
      y: b * (Math.cos(th) - 0.5 * e * dw(th)),
    });
    return {
      X, a, b,
      // point on the hand loop; side = +1 right, -1 left (mirror image)
      P: (th, side) => ({
        x: side * (X + a * (Math.cos(th) + q * e * w(th))),
        y: b * (Math.sin(th) - 0.5 * e * w(th)),
      }),
      // direction of travel (clockwise for the right hand, mirrored for left)
      T: (th, side) => { const d = dP(th); return { x: -side * d.x, y: -d.y }; },
    };
  }

  // Two conditions for a gravity parabola from P(thT) on the right hand to
  // P(thC) on the catching hand, tangent to both ellipses, with flight F:
  //   chord slope = mean of end slopes   (true of every parabola)
  //   dx * (s0 - s1) / g = F^2           (fixes horizontal speed vs. gravity)
  function residual(geom, thT, thC, cross, F) {
    const pT = geom.P(thT, 1), dT = geom.T(thT, 1);
    const side = cross ? -1 : 1;
    const pC = geom.P(thC, side), dC = geom.T(thC, side);
    const s0 = dT.y / dT.x, s1 = dC.y / dC.x;
    const dx = pC.x - pT.x, dy = pC.y - pT.y;
    return [
      (dy - (s0 + s1) * dx / 2) / Math.abs(dx),
      (dx * (s0 - s1) / G - F * F) / (F * F),
    ];
  }

  function solveThrow(geom, cross, F) {
    const eps = 1e-4;
    // Throw: lower-inner quadrant for a cascade (moving up and across),
    // upper-inner for a fountain (moving up and slightly out).
    // Catch: upper-outer quadrant (moving down and out).
    const tLo = cross ? Math.PI + eps : Math.PI / 2 + eps;
    const tHi = cross ? 1.5 * Math.PI - eps : Math.PI - eps;
    const cLo = eps, cHi = Math.PI / 2 - eps;
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    const norm = (r) => Math.hypot(r[0], r[1]);

    // coarse grid, then Newton on the best cell
    let best = null;
    const N = 48;
    for (let i = 0; i <= N; i++) {
      for (let j = 0; j <= N; j++) {
        const thT = tLo + (tHi - tLo) * i / N;
        const thC = cLo + (cHi - cLo) * j / N;
        const e = norm(residual(geom, thT, thC, cross, F));
        if (!best || e < best.e) best = { thT, thC, e };
      }
    }

    let thT = best.thT, thC = best.thC;
    for (let it = 0; it < 60; it++) {
      const r = residual(geom, thT, thC, cross, F);
      if (norm(r) < 1e-12) break;
      const h = 1e-7;
      const rT = residual(geom, thT + h, thC, cross, F);
      const rC = residual(geom, thT, thC + h, cross, F);
      const J00 = (rT[0] - r[0]) / h, J10 = (rT[1] - r[1]) / h;
      const J01 = (rC[0] - r[0]) / h, J11 = (rC[1] - r[1]) / h;
      const det = J00 * J11 - J01 * J10;
      if (Math.abs(det) < 1e-14) break;
      let dT = -(J11 * r[0] - J01 * r[1]) / det;
      let dC = -(-J10 * r[0] + J00 * r[1]) / det;
      const step = Math.hypot(dT, dC);
      if (step > 0.1) { dT *= 0.1 / step; dC *= 0.1 / step; }
      // backtrack so the residual never grows
      let k = 1;
      while (k > 1e-4) {
        const nT = clamp(thT + k * dT, tLo, tHi), nC = clamp(thC + k * dC, cLo, cHi);
        if (norm(residual(geom, nT, nC, cross, F)) < norm(r)) { thT = nT; thC = nC; break; }
        k /= 2;
      }
      if (k <= 1e-4) break;
    }
    return { thT, thC, err: norm(residual(geom, thT, thC, cross, F)) };
  }

  // Fallback for makeSegment below: distance covered over a segment of
  // duration T and length A, starting at speed w0 and ending at w1. Linear-ish
  // blend of the end speeds plus a smooth bump that carries the rest. The
  // exponent p sharpens the end contributions when the hand must move much
  // faster than its average, so the speed never goes negative.
  function makeSharpSegment(T, A, w0, w1) {
    const avg = A / T;
    const p = Math.max(1, (w0 + w1) / (0.75 * avg) - 1);
    const c = avg - (w0 + w1) / (p + 1);
    const q = p + 1;
    return (t) => {
      const u = Math.min(1, Math.max(0, t / T));
      return T * (w0 * (1 - Math.pow(1 - u, q)) / q + w1 * Math.pow(u, q) / q + c * (3 * u * u - 2 * u * u * u));
    };
  }

  // Distance covered over a segment of duration T and length A, starting at
  // speed w0 and ending at w1, with the speed changing as gradually as it
  // can. The speed is r(u)^2 with r quadratic: r runs from sqrt(w0) to
  // sqrt(w1) and its curvature m is the smallest that makes the distance
  // come out right. Squaring keeps the speed from going negative, and a
  // single smooth dip (or rise) spreads the slowing-down over the whole
  // segment instead of dumping it right after the release. Falls back to
  // the sharper profile only if no such dip stays positive.
  function makeSegment(T, A, w0, w1) {
    const avg = A / T;
    const r0 = Math.sqrt(w0), r1 = Math.sqrt(w1);
    // integral of r^2 over [0,1] as a quadratic in m
    const qa = 1 / 30, qb = (r0 + r1) / 6, qc = (r0 * r0 + r0 * r1 + r1 * r1) / 3 - avg;
    const disc = qb * qb - 4 * qa * qc;
    if (disc < 0) return makeSharpSegment(T, A, w0, w1);
    const m = (-qb + Math.sqrt(disc)) / (2 * qa);
    // r(u) = al + be*u + ga*u^2
    const al = r0, be = r1 - r0 + m, ga = -m;
    const uStar = ga !== 0 ? -be / (2 * ga) : -1;
    if (uStar > 0 && uStar < 1 && al + be * uStar + ga * uStar * uStar < 0) {
      return makeSharpSegment(T, A, w0, w1);
    }
    return (t) => {
      const u = Math.min(1, Math.max(0, t / T));
      const u2 = u * u, u3 = u2 * u;
      return T * (al * al * u + al * be * u2 + (be * be + 2 * al * ga) * u3 / 3 +
        be * ga * u3 * u / 2 + ga * ga * u3 * u2 / 5);
    };
  }

  // The empty return swing, when the hand leaves the throw much faster than
  // it needs to average. It slows gradually while covering most of the way
  // to the catch, hovers just short of it, then drops away with the
  // arriving ball:
  //   speed = w0 (1 - u/u1)^2          brake, over the first part
  //         + c 6u(1-u)                a slow creep throughout
  //         + w1 ((u - u2)/(1 - u2))^2 drop into the catch, at the end
  // Each piece meets zero with zero slope, so the speed stays smooth. The
  // drop and the creep take fixed shares of the distance and the brake
  // length u1 absorbs the rest. Returns null when the swing isn't fast
  // enough at its ends for this shape to fit.
  function makeReturnSegment(T, A, w0, w1) {
    const avg = A / T;
    const creep = 0.08;                                // share of distance
    const rho = Math.min(0.3, 3 * avg * 0.25 / w1);   // drop: <= 25% of distance
    const u2 = 1 - rho;
    const u1 = 3 * (avg * (1 - creep) - w1 * rho / 3) / w0;
    if (!(u1 > 0.2 && u1 < u2)) return null;
    const c = avg * creep;
    return (t) => {
      const u = Math.min(1, Math.max(0, t / T));
      const brake = u < u1 ? (u1 / 3) * (1 - Math.pow(1 - u / u1, 3)) : u1 / 3;
      const drop = u > u2 ? (rho / 3) * Math.pow((u - u2) / rho, 3) : 0;
      return T * (w0 * brake + w1 * drop + c * (3 * u * u - 2 * u * u * u));
    };
  }

  // ---------- model ----------

  function buildModel(opts) {
    const { n, beat: B } = opts;
    // A ball can't jump from the catch point to the throw point in zero
    // time, so "no dwell" means the shortest carry: a quick hairpin.
    const D = Math.max(MIN_DWELL, opts.dwell);
    const pattern = patternFor(n);
    const h = Math.max(...pattern);
    const cross = h % 2 === 1;
    const geom = makeGeom(opts.spacing / 2, opts.width / 2, opts.depth / 2, BULGE, opts.reverse ? 1 : -1);
    const F = (h - D) * B;

    const { thT, thC, err } = solveThrow(geom, cross, F);
    const pT = geom.P(thT, 1);
    const pC = geom.P(thC, cross ? -1 : 1);
    const dT = geom.T(thT, 1);
    const vx = (pC.x - pT.x) / F;
    const v0 = { x: vx, y: vx * dT.y / dT.x };
    const v1 = { x: vx, y: v0.y - G * F };
    const apex = pT.y + (v0.y > 0 ? v0.y * v0.y / (2 * G) : 0);

    // paddle face directions: along the launch, and back up the arrival
    // (arrival at the right hand is the mirror image when the throw crosses)
    const vc = cross ? { x: -v1.x, y: v1.y } : v1;
    const nT = Math.atan2(v0.y, v0.x);
    const nC = Math.atan2(-vc.y, -vc.x);

    // The hand is timed by distance along its loop, so its speed profile is
    // its real speed. arc[i] is the length from the catch point, going
    // clockwise, to angle thC - TAU*i/N.
    const N = 2048, arc = new Float64Array(N + 1);
    for (let i = 1; i <= N; i++) {
      const p = geom.P(thC - TAU * (i - 1) / N, 1), q = geom.P(thC - TAU * i / N, 1);
      arc[i] = arc[i - 1] + Math.hypot(q.x - p.x, q.y - p.y);
    }
    const arcAt = (phi) => {
      const f = phi / TAU * N, i = Math.min(N - 1, Math.floor(f));
      return arc[i] + (arc[i + 1] - arc[i]) * (f - i);
    };
    const A1 = mod(thC - thT, TAU);           // clockwise angle, catch -> throw (holding)
    const L1 = arcAt(A1);                     // ... and its length
    const L2 = arc[N] - L1;                   // throw -> catch (empty)
    const Td = D * B, Te = (2 - D) * B;
    // hand speeds that match the ball exactly at catch and throw
    const seg1 = makeSegment(Td, L1, Math.hypot(v1.x, v1.y), Math.hypot(v0.x, v0.y));
    const seg2 = makeReturnSegment(Te, L2, Math.hypot(v0.x, v0.y), Math.hypot(v1.x, v1.y)) ||
      makeSegment(Te, L2, Math.hypot(v0.x, v0.y), Math.hypot(v1.x, v1.y));

    // Ball bookkeeping: simulate throws from beat 0, a ball thrown at beat b
    // with height k lands at beat b+k. Ids are steady after the first cycle.
    const L = pattern.length;
    const M = L % 2 === 0 ? L : 2 * L;
    const Q = M * n;                      // every ball's period divides Q
    const lands = new Map();
    const events = [];                    // events[b] = ball id thrown at beat b, or -1
    let nextId = 0;
    for (let b = 0; b < 4 * Q; b++) {
      const k = pattern[b % L];
      if (k > 0) {
        let id = lands.get(b);
        if (id === undefined) id = nextId++;
        lands.set(b + k, id);
        events[b] = id;
      } else events[b] = -1;
    }

    // overall size of the ball path: hand loops at the bottom, peak on top
    let xMax = 0, yMin = 0;
    for (let i = 0; i < 128; i++) {
      const p = geom.P(TAU * i / 128, 1);
      xMax = Math.max(xMax, p.x); yMin = Math.min(yMin, p.y);
    }
    const extent = { w: 2 * xMax, h: Math.max(apex, geom.b) - yMin, xMax, yMin };

    return {
      n, h, pattern, cross, geom, D, B, F, thT, thC, pT, pC, v0, v1, apex, err,
      extent,
      A1, L1, L2, arc, Td, Te, seg1, seg2, Q, events, L, nT, nC,
    };
  }

  // hand angle at time tau since its last catch, tau in [0, 2B)
  function handAngle(m, tau) {
    const s = tau < m.Td ? m.seg1(tau) : m.L1 + m.seg2(tau - m.Td);
    // invert the arc-length table
    const arc = m.arc, N = arc.length - 1;
    let lo = 0, hi = N;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (arc[mid] <= s) lo = mid; else hi = mid; }
    const f = lo + Math.min(1, Math.max(0, (s - arc[lo]) / (arc[hi] - arc[lo] || 1)));
    return m.thC - TAU * f / N;
  }

  // right hand throws on even beats, left on odd
  function handPos(m, t, side) {
    const beats = t / m.B - (side === 1 ? 0 : 1) + m.D;
    const tau = mod(beats, 2) * m.B;
    return m.geom.P(handAngle(m, tau), side);
  }

  // Direction the paddle face points. At the throw it points along the
  // launch velocity, at the catch it points back up the incoming velocity,
  // so the ball always leaves and lands square on the face. In between it
  // turns in step with the hand's progress round the ellipse.
  function handNormal(m, t, side) {
    const beats = t / m.B - (side === 1 ? 0 : 1) + m.D;
    const tau = mod(beats, 2) * m.B;
    const wrap = (a) => mod(a + Math.PI, TAU) - Math.PI;
    const ang = tau < m.Td
      ? m.nC + wrap(m.nT - m.nC) * m.seg1(tau) / m.L1
      : m.nT + wrap(m.nC - m.nT) * m.seg2(tau - m.Td) / m.L2;
    return { x: side * Math.cos(ang), y: Math.sin(ang) };
  }

  function flightPos(m, tau, side) {
    return {
      x: side * (m.pT.x + m.v0.x * tau),
      y: m.pT.y + m.v0.y * tau - 0.5 * G * tau * tau,
    };
  }

  // positions of every ball at time t (standard sense)
  function ballsAt(m, t) {
    const tb = mod(t / m.B, m.Q) + 2 * m.Q;   // steady-state window
    const out = [];
    const top = Math.floor(tb);
    for (let b = top; b > top - m.h - 1; b--) {
      const id = m.events[b];
      if (id === undefined || id < 0) continue;
      const k = m.pattern[b % m.L];
      if (tb >= b + k) continue;
      const tau = (tb - b) * m.B;
      const throwSide = b % 2 === 0 ? 1 : -1;
      const catchSide = (b + k) % 2 === 0 ? 1 : -1;
      const pos = tau < m.F ? flightPos(m, tau, throwSide) : handPos(m, t, catchSide);
      out.push({ id, ...pos, held: tau >= m.F });
    }
    return out;
  }

  if (typeof window === 'undefined') {
    module.exports = { buildModel, ballsAt, handPos, G };
    return;
  }

  // ---------- UI / render ----------

  const canvas = document.getElementById('stage');
  const ctx = canvas.getContext('2d');
  const $ = (id) => document.getElementById(id);

  const opts = { n: 3, reverse: false, dwell: 0.6, beat: 0.25, rate: 0.5, spacing: 0.3, width: 0.22, depth: 0.14, guides: true, trails: true };
  let model = buildModel(opts);
  let simT = 0, paused = false, last = performance.now();
  let view = null;

  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const palette = ['#e4572e', '#2e86ab', '#f3a712', '#4caf50', '#a23b72', '#29335c', '#ff7f50', '#17bebb', '#8e6c8a'];

  function rebuild() {
    opts.dwell = Math.min(opts.dwell, 1.9);
    // keep the two hand loops from overlapping in the middle
    const width = Math.min(opts.width, opts.spacing - 0.02);
    model = buildModel({ ...opts, width });
    $('count').textContent = opts.n;
    $('style').textContent = opts.reverse ? 'outside' : 'inside';
    $('dwellVal').textContent = Math.round(opts.dwell / 2 * 100) + '%';
    $('beatVal').textContent = opts.beat.toFixed(2) + 's';
    $('rateVal').textContent = opts.rate.toFixed(2) + '×';
    const cm = (v) => Math.round(v * 100) + 'cm';
    $('spacingVal').textContent = cm(opts.spacing);
    $('widthVal').textContent = cm(width);
    $('depthVal').textContent = cm(opts.depth);
    const kind = model.cross ? 'cascade' : 'fountain';
    const name = model.cross ? (opts.reverse ? 'reverse ' : '') + kind : kind + (opts.reverse ? ' (reverse)' : '');
    $('stats').textContent =
      `${name}  ${model.pattern.join('')}\n` +
      `flight ${model.F.toFixed(2)}s  peak ${(model.apex * 100).toFixed(0)}cm
` +
      `shape ${cm(model.extent.w)} wide × ${cm(model.extent.h)} tall`;
  }

  $('fewer').onclick = () => { opts.n = Math.max(1, opts.n - 1); rebuild(); };
  $('more').onclick = () => { opts.n = Math.min(9, opts.n + 1); rebuild(); };
  $('style').onclick = () => { opts.reverse = !opts.reverse; rebuild(); };
  $('dwell').oninput = (e) => { opts.dwell = +e.target.value; rebuild(); };
  $('spacing').oninput = (e) => { opts.spacing = +e.target.value; rebuild(); };
  $('width').oninput = (e) => { opts.width = +e.target.value; rebuild(); };
  $('depth').oninput = (e) => { opts.depth = +e.target.value; rebuild(); };
  $('beat').oninput = (e) => { opts.beat = +e.target.value; rebuild(); };
  $('rate').oninput = (e) => { opts.rate = +e.target.value; rebuild(); };
  $('guides').onchange = (e) => { opts.guides = e.target.checked; };
  $('trails').onchange = (e) => { opts.trails = e.target.checked; };
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' && e.target.type === 'range') return;
    if (e.key === ' ') { paused = !paused; e.preventDefault(); }
    else if (e.key >= '1' && e.key <= '9') { opts.n = +e.key; rebuild(); }
    else if (e.key === 'ArrowRight') { paused = true; simT += 1 / 60; }
    else if (e.key === 'ArrowLeft') { paused = true; simT -= 1 / 60; }
  });

  function resize() {
    const dpr = devicePixelRatio || 1;
    canvas.width = innerWidth * dpr;
    canvas.height = innerHeight * dpr;
  }
  addEventListener('resize', resize);
  resize();

  function targetView() {
    const g = model.geom;
    const xMax = model.extent.xMax + BALL_R + 0.06;
    const yMin = model.extent.yMin - BALL_R - 0.06;
    const yMax = Math.max(model.apex, g.b) + BALL_R + 0.05;
    const w = canvas.width, h = canvas.height;
    const pad = 0.08;
    const s = Math.min(w * (1 - 2 * pad) / (2 * xMax), h * (1 - 2 * pad) / (yMax - yMin));
    return { s, cy: (yMax + yMin) / 2 };
  }

  function updateView(dt) {
    const t = targetView();
    if (!view) { view = t; return; }
    const k = 1 - Math.exp(-dt * 4);
    view.s += (t.s - view.s) * k;
    view.cy += (t.cy - view.cy) * k;
  }

  const sx = (x) => canvas.width / 2 + x * view.s;
  const sy = (y) => canvas.height / 2 - (y - view.cy) * view.s;

  function drawGuides() {
    const m = model, g = m.geom, dpr = devicePixelRatio || 1;
    ctx.lineWidth = 1.2 * dpr;

    // full hand ellipses, dashed
    ctx.strokeStyle = color('--guide');
    ctx.setLineDash([4 * dpr, 5 * dpr]);
    for (const side of [1, -1]) {
      ctx.beginPath();
      for (let i = 0; i <= 160; i++) {
        const p = g.P(TAU * i / 160, side);
        i === 0 ? ctx.moveTo(sx(p.x), sy(p.y)) : ctx.lineTo(sx(p.x), sy(p.y));
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // ball path: dwell arcs + flight parabolas
    ctx.strokeStyle = color('--path');
    ctx.beginPath();
    for (const side of [1, -1]) {
      const steps = 80;
      for (let i = 0; i <= steps; i++) {
        const p = g.P(m.thC - m.A1 * i / steps, side);
        i === 0 ? ctx.moveTo(sx(p.x), sy(p.y)) : ctx.lineTo(sx(p.x), sy(p.y));
      }
      for (let i = 0; i <= steps; i++) {
        const p = flightPos(m, m.F * i / steps, side);
        i === 0 ? ctx.moveTo(sx(p.x), sy(p.y)) : ctx.lineTo(sx(p.x), sy(p.y));
      }
    }
    ctx.stroke();

    // tangency points: filled = throw, hollow = catch
    const r = 3 * dpr;
    for (const side of [1, -1]) {
      const pThrow = g.P(opts.reverse ? m.thC : m.thT, side);
      const pCatch = g.P(opts.reverse ? m.thT : m.thC, side);
      ctx.fillStyle = color('--path');
      ctx.beginPath(); ctx.arc(sx(pThrow.x), sy(pThrow.y), r, 0, TAU); ctx.fill();
      ctx.strokeStyle = color('--path');
      ctx.beginPath(); ctx.arc(sx(pCatch.x), sy(pCatch.y), r, 0, TAU); ctx.stroke();
    }
  }

  function drawPaddle(t, side) {
    const m = model;
    const p = handPos(m, t, side);
    const { x: nx, y: ny } = handNormal(m, t, side);
    const thick = 0.01, half = 0.04;
    const off = BALL_R + thick / 2;
    const cx = p.x - nx * off, cy = p.y - ny * off;
    const ux = ny, uy = -nx;
    ctx.strokeStyle = color('--ink');
    ctx.lineWidth = thick * view.s;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(sx(cx - ux * half), sy(cy - uy * half));
    ctx.lineTo(sx(cx + ux * half), sy(cy + uy * half));
    ctx.stroke();
  }

  // The motion is a closed-form function of time, so a trail is just the
  // same balls sampled a little in the past; no history buffer needed.
  function drawTrails() {
    const steps = 120, span = 0.35 * model.B;   // fine enough for the fast swing through the bottom
    const trails = new Map();
    for (let i = steps; i >= 0; i--) {
      const s = simT - span * i / steps;
      for (const b of ballsAt(model, opts.reverse ? -s : s)) {
        if (!trails.has(b.id)) trails.set(b.id, []);
        trails.get(b.id).push(b);
      }
    }
    ctx.lineCap = 'butt';   // round caps overlap and look beaded at low alpha
    for (const [id, pts] of trails) {
      ctx.strokeStyle = palette[id % palette.length];
      for (let i = 1; i < pts.length; i++) {
        const f = i / (pts.length - 1);           // 0 oldest .. 1 newest
        ctx.globalAlpha = 0.3 * f * f;
        ctx.lineWidth = 2 * BALL_R * view.s;
        ctx.beginPath();
        ctx.moveTo(sx(pts[i - 1].x), sy(pts[i - 1].y));
        ctx.lineTo(sx(pts[i].x), sy(pts[i].y));
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!paused) simT += dt * opts.rate;
    updateView(dt);

    ctx.fillStyle = color('--bg');
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (opts.guides) drawGuides();

    const t = opts.reverse ? -simT : simT;
    drawPaddle(t, 1);
    drawPaddle(t, -1);
    if (opts.trails) drawTrails();

    for (const ball of ballsAt(model, t)) {
      ctx.fillStyle = palette[ball.id % palette.length];
      ctx.beginPath();
      ctx.arc(sx(ball.x), sy(ball.y), BALL_R * view.s, 0, TAU);
      ctx.fill();
    }
    requestAnimationFrame(frame);
  }

  rebuild();
  requestAnimationFrame(frame);
})();
