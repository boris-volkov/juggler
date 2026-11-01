// Juggling simulation.
//
// World units are metres and seconds, y points up, gravity is real (9.81).
// Each hand moves around a loop. The ball path is smooth everywhere: every
// flight parabola leaves one hand loop tangentially (same direction and
// speed as the hand) and arrives at the next loop tangentially, then rides
// the bottom of that loop until the next throw. (CATCH_GIVE below 1 lets the
// hand meet a ball slower than it falls, with the impact taking up the
// difference.) In a cascade the two parabolas plus the two carries form the
// figure-8; the loops touch it along its two low points.
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
  const CATCH_GIVE = 1;      // hand speed at a catch, as a share of the ball's
  const BULGE = 0.3;         // bottom bulges of the hand loops (throw and catch sides)
  const TAU = Math.PI * 2;
  const REST_TH = -Math.PI / 2;  // bottom of the hand loop, where an idle hand waits

  const mod = (x, m) => ((x % m) + m) % m;

  // 1 and 2 balls reuse the 3-ball rhythm with gaps so the figure-8 survives.
  function patternFor(n) {
    if (n === 1) return [3, 0, 0];
    if (n === 2) return [3, 3, 0];
    return [n];
  }

  // ---------- geometry ----------

  // Hand loop: an ellipse with both bottom quarters pushed out and down, like
  // a racing line swinging wide through a U-turn. One bulge sits on the
  // throwing side, so the hand has a longer straight run to accelerate along
  // the throw; the other on the catching side, so it can carry on with the
  // ball and slow it before turning. The turn itself is left for the bottom.
  // Each push (sg = +1 outside, -1 inside) is
  //   w = sin^2(th) * (1 + sg cos th) / 2   (lower half only)
  // which peaks below the centre on its side, fades toward the other, and is
  // zero with zero slope at the centre line, so the sides keep their vertical
  // tangents and the joint with the round upper half is smooth.
  //
  // The upper outside quarter is then pulled up by `rise` metres with the
  // same kind of push, mirrored into the upper half. That lengthens the empty
  // hand's swing, so it can leave a throw and meet a catch at the ball's own
  // speed without stalling in between. Throwing inside, it is the hand
  // reaching up to meet the ball (and catching it a little higher); throwing
  // outside, the same shape run backwards, it is the follow-through.
  function makeGeom(X, a, b, e, rise) {
    const w = (th, sg) => {
      const s = Math.sin(th);
      return s < 0 ? s * s * (1 + sg * Math.cos(th)) / 2 : 0;
    };
    const dw = (th, sg) => {
      const s = Math.sin(th), c = Math.cos(th);
      return s < 0 ? s * c * (1 + sg * c) - sg * s * s * s / 2 : 0;
    };
    // net push of the two bulges: sideways (outside minus inside) and down
    const px = (th) => w(th, 1) - w(th, -1), py = (th) => w(th, 1) + w(th, -1);
    const dpx = (th) => dw(th, 1) - dw(th, -1), dpy = (th) => dw(th, 1) + dw(th, -1);
    // the catching side's rise: w mirrored into the upper half
    const up = (th) => w(-th, 1), dup = (th) => -dw(-th, 1);
    const dP = (th) => ({
      x: a * (-Math.sin(th) + e * dpx(th)),
      y: b * (Math.cos(th) - 0.5 * e * dpy(th)) + rise * dup(th),
    });
    return {
      X, a, b,
      // point on the hand loop; side = +1 right, -1 left (mirror image)
      P: (th, side) => ({
        x: side * (X + a * (Math.cos(th) + e * px(th))),
        y: b * (Math.sin(th) - 0.5 * e * py(th)) + rise * up(th),
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
    // the ball must leave and arrive moving the same way as the hands
    if (Math.sign(dT.x) !== Math.sign(dx) || Math.sign(dC.x) !== Math.sign(dx)) return [1e3, 1e3];
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

  // ---------- model ----------

  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const lcm = (a, b) => a / gcd(a, b) * b;

  // A siteswap is valid when its throws land on distinct beats, which also
  // makes the average a whole number: the number of balls. 1s (a ball
  // handed straight across) aren't supported yet: they have no time to fly.
  function parseSiteswap(text) {
    const s = text.trim();
    if (!/^[0-9]+$/.test(s)) return { error: 'use digits 0-9' };
    const p = [...s].map(Number), L = p.length;
    if (p.includes(1)) return { error: '1s (hand-across passes) not supported yet' };
    if (Math.max(...p) < 3) return { error: 'needs a throw of 3 or more' };
    const lands = new Set(p.map((k, i) => (i + k) % L));
    if (lands.size !== L) return { error: 'throws collide' };
    const sum = p.reduce((a, b) => a + b, 0);
    if (sum % L) return { error: 'not a whole number of balls' };
    return { pattern: p, n: sum / L };
  }

  function buildModel(opts) {
    const B = opts.beat;
    // A ball can't jump from the catch point to the throw point in zero
    // time, so "no dwell" means the shortest carry: a quick hairpin.
    const D = Math.max(MIN_DWELL, opts.dwell);
    // Throwing outside is shown as the model run backwards in time. Run
    // backwards, a throw from beat b landing at b + k becomes a throw of the
    // same height from beat -(b + k), so build that pattern then.
    const L = opts.pattern.length;
    let pattern = opts.pattern;
    if (opts.reverse) {
      pattern = new Array(L).fill(0);
      opts.pattern.forEach((k, i) => { if (k > 0) pattern[mod(-(i + k), L)] = k; });
    }
    const n = pattern.reduce((a, b) => a + b, 0) / L;
    const h = Math.max(...pattern);
    const geom = makeGeom(opts.spacing / 2, opts.width / 2, opts.depth / 2, BULGE, opts.reach);

    // One tangent flight for each throw height of 3 or more: odd heights
    // cross to the other hand, even ones come back to the thrower. Solved
    // for the right hand; the left is its mirror image.
    const types = {};
    let err = 0, apex = -Infinity;
    for (const k of new Set(pattern)) {
      if (k < 3) continue;
      const cross = k % 2 === 1, F = (k - D) * B;
      const sol = solveThrow(geom, cross, F);
      const pT = geom.P(sol.thT, 1);
      const pC = geom.P(sol.thC, cross ? -1 : 1);
      const dT = geom.T(sol.thT, 1);
      const vx = (pC.x - pT.x) / F;
      const v0 = { x: vx, y: vx * dT.y / dT.x };
      const v1 = { x: vx, y: v0.y - G * F };
      // arrival at the right hand is the mirror image when the throw crosses
      const vc = cross ? { x: -v1.x, y: v1.y } : v1;
      const top = pT.y + (v0.y > 0 ? v0.y * v0.y / (2 * G) : 0);
      types[k] = {
        k, cross, F, thT: sol.thT, thC: sol.thC, pT, v0, v1, apex: top,
        // paddle face directions: along the launch, and back up the arrival
        nT: Math.atan2(v0.y, v0.x), nC: Math.atan2(-vc.y, -vc.x),
        // Hand speeds at the ends of a carry. A throw has to match the ball:
        // it leaves at the hand's speed. A catch needn't: the hand meets the
        // ball at CATCH_GIVE of its speed and the rest goes in the impact.
        // In the reverse pattern the standard throw point is where catches
        // happen, so the give swaps ends.
        sT: Math.hypot(v0.x, v0.y) * (opts.reverse ? CATCH_GIVE : 1),
        sC: Math.hypot(v1.x, v1.y) * (opts.reverse ? 1 : CATCH_GIVE),
      };
      err = Math.max(err, sol.err);
      apex = Math.max(apex, top);
    }

    // Hands are timed by distance along their loop, so a speed profile is a
    // real speed. arc[i] is the length going clockwise from angle 0 to angle
    // -TAU*i/N, over two turns so a stretch never has to wrap.
    const N = 2048, arc = new Float64Array(2 * N + 1);
    for (let i = 1; i <= 2 * N; i++) {
      const p = geom.P(-TAU * (i - 1) / N, 1), q = geom.P(-TAU * i / N, 1);
      arc[i] = arc[i - 1] + Math.hypot(q.x - p.x, q.y - p.y);
    }
    const lenAt = (phi) => {
      const f = phi / TAU * N, i = Math.min(2 * N - 1, Math.floor(f));
      return arc[i] + (arc[i + 1] - arc[i]) * (f - i);
    };

    // Each hand works through a timeline of keyframes. At each of its beats
    // it throws (a height of 3 or more), or holds its ball (a 2) or is empty
    // (a 0), and then rests at the bottom of its loop. Before a beat, it
    // catches whatever was thrown to land then. Between keyframes it moves
    // clockwise round the loop with a smooth speed profile.
    const landing = new Array(L).fill(0);   // height of the throw landing on each beat
    pattern.forEach((k, i) => { if (k > 0) landing[(i + k) % L] = k; });
    const P = L % 2 === 0 ? L : 2 * L;      // beats before both hands repeat
    const hands = [1, -1].map((side) => {
      const kf = [];
      for (let b = side === 1 ? 0 : 1; b < P; b += 2) {
        const k = pattern[b % L], kin = landing[b % L];
        if (kin >= 3) {
          const ty = types[kin];
          kf.push({ t: (b - D) * B, th: ty.thC, v: ty.sC, nrm: ty.nC, hold: true });
        }
        if (k >= 3) {
          const ty = types[k];
          kf.push({ t: b * B, th: ty.thT, v: ty.sT, nrm: ty.nT, hold: false });
        } else {
          kf.push({ t: b * B, th: REST_TH, v: 0, nrm: Math.PI / 2, hold: k === 2 });
        }
      }
      const period = P * B;
      kf.forEach((f) => { f.t = mod(f.t, period); });
      kf.sort((a, b) => a.t - b.t);
      const segs = kf.map((a, i) => {
        const z = kf[(i + 1) % kf.length];
        const T = mod(z.t - a.t, period) || period;
        const phi0 = mod(-a.th, TAU), dphi = mod(a.th - z.th, TAU);
        const s0 = lenAt(phi0), len = lenAt(phi0 + dphi) - s0;
        const wrap = (x) => mod(x + Math.PI, TAU) - Math.PI;
        return {
          t0: a.t, T, s0, len, hold: a.hold,
          prof: makeSegment(T, len, a.v, z.v),
          n0: a.nrm, dn: wrap(z.nrm - a.nrm),
        };
      });
      return { segs, period };
    });

    // Ball bookkeeping: simulate throws from beat 0, a ball thrown at beat b
    // with height k lands at beat b+k. Every ball's round trip is a multiple
    // of Q beats, so ids are periodic in Q once the first balls are in.
    let Q = 1;
    for (let r = 0; r < P; r++) {
      if (pattern[r % L] === 0) continue;
      let c = r, T = 0, steps = 0;
      do { T += pattern[c % L]; c = (c + pattern[c % L]) % P; } while (c !== r && ++steps < P);
      Q = lcm(Q, T);
    }
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

    return { n, h, pattern, geom, D, B, types, apex, err, extent, arc, hands, Q, events, L };
  }

  // the segment of a hand's timeline running at time t, and the time into it
  function handSeg(m, t, side) {
    const hand = m.hands[side === 1 ? 0 : 1];
    const tp = mod(t, hand.period), segs = hand.segs;
    let i = segs.length - 1;
    while (i >= 0 && segs[i].t0 > tp) i--;
    const seg = segs[i < 0 ? segs.length - 1 : i];   // before the first keyframe: the last, wrapping
    return { seg, dt: mod(tp - seg.t0, hand.period) };
  }

  function handPos(m, t, side) {
    const { seg, dt } = handSeg(m, t, side);
    const s = seg.s0 + seg.prof(dt);
    // invert the arc-length table
    const arc = m.arc, N = (arc.length - 1) / 2;
    let lo = 0, hi = arc.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (arc[mid] <= s) lo = mid; else hi = mid; }
    const f = lo + Math.min(1, Math.max(0, (s - arc[lo]) / (arc[hi] - arc[lo] || 1)));
    return m.geom.P(-TAU * f / N, side);
  }

  // Direction the paddle face points. At a throw it points along the launch
  // velocity, at a catch back up the incoming velocity, so the ball always
  // leaves and lands square on the face; resting, straight up. In between it
  // turns in step with the hand's progress along its loop.
  function handNormal(m, t, side) {
    const { seg, dt } = handSeg(m, t, side);
    const f = seg.len > 1e-9 ? seg.prof(dt) / seg.len : dt / seg.T;
    const ang = seg.n0 + seg.dn * f;
    return { x: side * Math.cos(ang), y: Math.sin(ang) };
  }

  function flightPos(ty, tau, side) {
    return {
      x: side * (ty.pT.x + ty.v0.x * tau),
      y: ty.pT.y + ty.v0.y * tau - 0.5 * G * tau * tau,
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
      // a 2 never leaves the hand
      const ty = m.types[k];
      const flying = ty && tau < ty.F;
      const pos = flying ? flightPos(ty, tau, throwSide) : handPos(m, t, catchSide);
      out.push({ id, ...pos, held: !flying });
    }
    return out;
  }

  // Throws and catches with display time s in (s0, s1]: { s, kind, id, side }.
  // Display time runs backwards through the model in the reverse pattern,
  // which turns each throw into a catch and each catch into a throw. Holds
  // (2s) make no sound.
  function eventsBetween(m, s0, s1, reverse) {
    const t0 = reverse ? -s1 : s0, t1 = reverse ? -s0 : s1;
    const out = [];
    const add = (t, kind, id, side) => {
      const s = reverse ? -t : t;
      if (s > s0 && s <= s1) {
        const k = reverse ? (kind === 'throw' ? 'catch' : 'throw') : kind;
        out.push({ s, kind: k, id, side });
      }
    };
    for (let b = Math.floor(t0 / m.B) - m.h - 1; b <= Math.ceil(t1 / m.B) + 1; b++) {
      const k = m.pattern[mod(b, m.L)];
      if (k < 3) continue;
      const id = m.events[mod(b, m.Q) + 2 * m.Q];
      add(b * m.B, 'throw', id, mod(b, 2) === 0 ? 1 : -1);
      add((b + k - m.D) * m.B, 'catch', id, mod(b + k, 2) === 0 ? 1 : -1);
    }
    return out.sort((a, b) => a.s - b.s);
  }

  if (typeof window === 'undefined') {
    module.exports = { buildModel, parseSiteswap, patternFor, ballsAt, handPos, handNormal, eventsBetween, G };
    return;
  }

  // ---------- UI / render ----------

  const canvas = document.getElementById('stage');
  const ctx = canvas.getContext('2d');
  const $ = (id) => document.getElementById(id);

  const opts = { n: 3, pattern: patternFor(3), reverse: false, dwell: 0.6, beat: 0.25, rate: 0.5, spacing: 0.3, width: 0.22, depth: 0.14, reach: 0.08, guides: true, trails: true, sound: false };
  let model = buildModel(opts);
  let simT = 0, paused = false, last = performance.now();
  let view = null;

  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const palette = ['#e4572e', '#2e86ab', '#f3a712', '#4caf50', '#a23b72', '#29335c', '#ff7f50', '#17bebb', '#8e6c8a'];

  // Browsers restore form controls on a reload but the script starts from
  // its defaults, so set the controls from the defaults to keep them agreeing.
  function syncControls() {
    const values = { dwell: opts.dwell, spacing: opts.spacing, width: opts.width,
      depth: opts.depth, rise: opts.reach, beat: opts.beat, rate: opts.rate };
    for (const [id, v] of Object.entries(values)) $(id).value = v;
    for (const id of ['guides', 'trails', 'sound']) $(id).checked = opts[id];
    $('siteswap').value = opts.pattern.join('');
  }

  function rebuild() {
    opts.dwell = Math.min(opts.dwell, 1.9);
    // keep the two hand loops from overlapping in the middle
    const width = Math.min(opts.width, opts.spacing - 0.02);
    model = buildModel({ ...opts, width });
    $('count').textContent = model.n;
    $('style').textContent = opts.reverse ? 'outside' : 'inside';
    $('dwellVal').textContent = Math.round(opts.dwell / 2 * 100) + '%';
    $('beatVal').textContent = opts.beat.toFixed(2) + 's';
    $('rateVal').textContent = opts.rate.toFixed(2) + '×';
    const cm = (v) => Math.round(v * 100) + 'cm';
    $('spacingVal').textContent = cm(opts.spacing);
    $('widthVal').textContent = cm(width);
    $('depthVal').textContent = cm(opts.depth);
    $('riseVal').textContent = cm(opts.reach);
    const ss = opts.pattern.join('');
    const uniform = opts.pattern.every((k) => k === opts.pattern[0]) || ss === '300' || ss === '330';
    const kind = Object.values(model.types).every((ty) => ty.cross) ? 'cascade' : 'fountain';
    const name = uniform ? (opts.reverse ? 'reverse ' : '') + kind : 'siteswap' + (opts.reverse ? ' (reverse)' : '');
    const peaks = Object.values(model.types).map((ty) => ty.apex * 100);
    const lo = Math.round(Math.min(...peaks)), hi = Math.round(Math.max(...peaks));
    $('stats').textContent =
      `${name}  ${ss}\n` +
      `peak ${lo === hi ? hi : lo + '–' + hi}cm\n` +
      `shape ${cm(model.extent.w)} wide × ${cm(model.extent.h)} tall`;
  }

  function setBalls(n) {
    opts.n = n;
    opts.pattern = patternFor(n);
    $('siteswap').value = opts.pattern.join('');
    $('siteswap').classList.remove('bad');
    rebuild();
  }
  function setSiteswap() {
    const box = $('siteswap'), r = parseSiteswap(box.value);
    box.classList.toggle('bad', !!r.error);
    box.title = r.error || '';
    if (r.error) return;
    opts.pattern = r.pattern;
    opts.n = r.n;
    rebuild();
  }
  $('fewer').onclick = () => setBalls(Math.max(1, model.n - 1));
  $('more').onclick = () => setBalls(Math.min(9, model.n + 1));
  $('siteswap').onchange = setSiteswap;
  $('siteswap').onkeydown = (e) => { if (e.key === 'Enter') e.target.blur(); };
  $('style').onclick = () => { opts.reverse = !opts.reverse; rebuild(); };
  $('dwell').oninput = (e) => { opts.dwell = +e.target.value; rebuild(); };
  $('spacing').oninput = (e) => { opts.spacing = +e.target.value; rebuild(); };
  $('width').oninput = (e) => { opts.width = +e.target.value; rebuild(); };
  $('depth').oninput = (e) => { opts.depth = +e.target.value; rebuild(); };
  $('rise').oninput = (e) => { opts.reach = +e.target.value; rebuild(); };
  $('beat').oninput = (e) => { opts.beat = +e.target.value; rebuild(); };
  $('rate').oninput = (e) => { opts.rate = +e.target.value; rebuild(); };
  $('guides').onchange = (e) => { opts.guides = e.target.checked; };
  $('trails').onchange = (e) => { opts.trails = e.target.checked; };
  $('sound').onchange = (e) => { opts.sound = e.target.checked; if (opts.sound) unlockAudio(); };
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox') return;
    if (e.key === ' ') { paused = !paused; e.preventDefault(); }
    else if (e.key >= '1' && e.key <= '9') setBalls(+e.key);
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
    // Fit the pattern clear of the control panel: in the space to its right
    // or the space below it, whichever lets the pattern be drawn larger
    // (beside it on a wide screen, under it on a phone held upright).
    const dpr = devicePixelRatio || 1;
    const r = $('panel').getBoundingClientRect();
    const gap = 16 * dpr, pad = 0.08;
    const fit = (x0, y0) => {
      const w = canvas.width - x0, h = canvas.height - y0;
      const s = Math.min(w * (1 - 2 * pad) / (2 * xMax), h * (1 - 2 * pad) / (yMax - yMin));
      return { s, cx: x0 + w / 2, cy: (yMax + yMin) / 2, oy: y0 / 2 };
    };
    const beside = fit(r.right * dpr + gap, 0), below = fit(0, r.bottom * dpr + gap);
    return beside.s >= below.s ? beside : below;
  }

  function updateView(dt) {
    const t = targetView();
    if (!view) { view = t; return; }
    const k = 1 - Math.exp(-dt * 4);
    view.s += (t.s - view.s) * k;
    view.cx += (t.cx - view.cx) * k;
    view.cy += (t.cy - view.cy) * k;
    view.oy += (t.oy - view.oy) * k;
  }

  const sx = (x) => view.cx + x * view.s;
  const sy = (y) => canvas.height / 2 + view.oy - (y - view.cy) * view.s;

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

    // ball path: each hand's carries (sampled from its own timeline, so holds
    // show too) and every kind of flight from either hand
    ctx.strokeStyle = color('--path');
    ctx.beginPath();
    for (const side of [1, -1]) {
      for (const seg of m.hands[side === 1 ? 0 : 1].segs) {
        if (!seg.hold || seg.len < 1e-6) continue;
        for (let i = 0; i <= 40; i++) {
          const p = handPos(m, seg.t0 + seg.T * i / 40, side);
          i === 0 ? ctx.moveTo(sx(p.x), sy(p.y)) : ctx.lineTo(sx(p.x), sy(p.y));
        }
      }
      for (const ty of Object.values(m.types)) {
        for (let i = 0; i <= 80; i++) {
          const p = flightPos(ty, ty.F * i / 80, side);
          i === 0 ? ctx.moveTo(sx(p.x), sy(p.y)) : ctx.lineTo(sx(p.x), sy(p.y));
        }
      }
    }
    ctx.stroke();

    // tangency points: filled = throw, hollow = catch
    const r = 3 * dpr;
    ctx.fillStyle = color('--path');
    for (const side of [1, -1]) {
      for (const ty of Object.values(m.types)) {
        const pThrow = g.P(opts.reverse ? ty.thC : ty.thT, side);
        const pCatch = g.P(opts.reverse ? ty.thT : ty.thC, side);
        ctx.beginPath(); ctx.arc(sx(pThrow.x), sy(pThrow.y), r, 0, TAU); ctx.fill();
        ctx.beginPath(); ctx.arc(sx(pCatch.x), sy(pCatch.y), r, 0, TAU); ctx.stroke();
      }
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

  // ---------- sound ----------
  //
  // Every throw and catch makes a sound, so the rhythm can be heard. A throw
  // is effort building to a release: a tone that swells and slides up from D
  // to G through the end of the carry, peaks the instant the ball leaves the
  // hand, then lets go. A catch is an impact: a short, soft C struck as the
  // ball lands. All three notes are from the C major pentatonic. Each sound
  // is panned toward the hand making it.
  // Browsers only start audio after a click or key press, so the context is
  // made on the first one.

  const THROW_NOTE = 392.0;    // G4, reached at the release
  const THROW_FROM = 293.66;   // D4, where the swell starts
  const CATCH_NOTE = 261.63;   // C4
  const SWELL_MAX = 0.25;      // longest throw swell (s)
  let audio = null;

  function unlockAudio() {
    if (!audio) {
      const ac = new (window.AudioContext || window.webkitAudioContext)();
      const master = ac.createGain();
      master.gain.value = 0.5;
      master.connect(ac.destination);
      audio = { ac, master };
    }
    if (audio.ac.state === 'suspended') audio.ac.resume();
  }
  addEventListener('pointerdown', unlockAudio);
  addEventListener('keydown', unlockAudio);

  // a soft mallet: a sine with a faint fourth-harmonic ping, quick attack,
  // exponential decay
  function ring(freq, gain, decay, pan) {
    const { ac, master } = audio, t = ac.currentTime;
    const env = ac.createGain(), panner = ac.createStereoPanner();
    panner.pan.value = pan;
    env.connect(panner).connect(master);
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(gain, t + 0.004);
    env.gain.exponentialRampToValueAtTime(0.0001, t + decay);
    for (const [mul, amp] of [[1, 1], [4, 0.12]]) {
      const osc = ac.createOscillator(), g = ac.createGain();
      osc.frequency.value = freq * mul;
      g.gain.value = amp;
      osc.connect(g).connect(env);
      osc.start(t);
      osc.stop(t + decay + 0.05);
    }
  }

  // a pure tone that swells from silence and slides up in pitch over `rise`
  // seconds to its peak, then falls away over `tail`
  function swell(from, to, gain, rise, tail, pan) {
    const { ac, master } = audio, t = ac.currentTime;
    const env = ac.createGain(), panner = ac.createStereoPanner(), osc = ac.createOscillator();
    panner.pan.value = pan;
    osc.connect(env).connect(panner).connect(master);
    osc.frequency.setValueAtTime(from, t);
    osc.frequency.exponentialRampToValueAtTime(to, t + rise);
    env.gain.setValueAtTime(0.0001, t);
    env.gain.exponentialRampToValueAtTime(gain, t + rise);
    env.gain.exponentialRampToValueAtTime(0.0001, t + rise + tail);
    osc.start(t);
    osc.stop(t + rise + tail + 0.05);
  }

  // Catches sound as the animation passes them. Throws are started early, by
  // the length of their swell, so the swell peaks on the release: the motion
  // is known ahead of time, so look that far ahead.
  function playEvents(s0, s1) {
    if (!opts.sound || !audio || audio.ac.state !== 'running') return;
    if (s1 <= s0 || s1 - s0 > 0.25) return;   // stepping back, or a long stall
    // the swell covers most of the carry, in real seconds
    const rise = Math.min(SWELL_MAX, 0.8 * model.D * model.B / opts.rate);
    const ahead = rise * opts.rate;
    for (const e of eventsBetween(model, s0, s1, opts.reverse)) {
      if (e.kind === 'catch') ring(CATCH_NOTE, 0.16, 0.3, 0.5 * e.side);
    }
    for (const e of eventsBetween(model, s0 + ahead, s1 + ahead, opts.reverse)) {
      if (e.kind === 'throw') {
        swell(THROW_FROM, THROW_NOTE, 0.2, Math.max(0.01, (e.s - s1) / opts.rate), 0.18, 0.5 * e.side);
      }
    }
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const prevT = simT;
    if (!paused) simT += dt * opts.rate;
    playEvents(prevT, simT);
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

  syncControls();
  rebuild();
  requestAnimationFrame(frame);
})();
