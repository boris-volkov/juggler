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
  const ONE_DWELL = 0.8;     // most dwell (beats) a pattern with 1s allows
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

  function solveThrow(geom, cross, F, k) {
    const eps = 1e-4;
    // Throw: lower-inner quadrant for a cascade (moving up and across),
    // upper-inner for a fountain (moving up and slightly out).
    // Catch: upper-outer quadrant (moving down and out). A 1 has too little
    // time to reach the far side: it is scooped up and across and caught
    // still rising, near the top of the other hand's loop.
    const tLo = cross ? Math.PI + eps : Math.PI / 2 + eps;
    const tHi = cross ? 1.5 * Math.PI - eps : Math.PI - eps;
    const cLo = eps, cHi = (k === 1 ? Math.PI : Math.PI / 2) - eps;
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

  // A hand's path between two keyframes when throwing in columns: from p0 to
  // p1, leaving and arriving vertically. h > 0 arches over (up, across, down:
  // the empty hand after a throw), h < 0 dips under (down, across, up: the
  // carry). w swings the middle sideways, so a path that starts and ends in
  // the same column turns round instead of stopping dead and reversing.
  // Returned as a function of distance along it, for arc-length timing.
  function makeArch(p0, p1, h, w) {
    const at = (f) => {
      const c = (1 - Math.cos(f)) / 2, s = Math.sin(f);
      return { x: p0.x + (p1.x - p0.x) * c + w * s * s, y: p0.y + (p1.y - p0.y) * c + h * s };
    };
    const N = 256, pts = [at(0)], acc = new Float64Array(N + 1);
    for (let i = 1; i <= N; i++) {
      pts.push(at(Math.PI * i / N));
      acc[i] = acc[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    }
    return {
      len: acc[N],
      at(s) {
        let lo = 0, hi = N;
        while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (acc[mid] <= s) lo = mid; else hi = mid; }
        const f = Math.min(1, Math.max(0, (s - acc[lo]) / (acc[hi] - acc[lo] || 1)));
        return { x: pts[lo].x + (pts[hi].x - pts[lo].x) * f, y: pts[lo].y + (pts[hi].y - pts[lo].y) * f };
      },
    };
  }

  // ---------- model ----------

  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const lcm = (a, b) => a / gcd(a, b) * b;

  // A siteswap is valid when its throws land on distinct beats, which also
  // makes the average a whole number: the number of balls.
  function parseSiteswap(text) {
    const s = text.trim();
    if (!/^[0-9]+$/.test(s)) return { error: 'use digits 0-9' };
    const p = [...s].map(Number), L = p.length;
    if (Math.max(...p) < 3) return { error: 'needs a throw of 3 or more' };
    const lands = new Set(p.map((k, i) => (i + k) % L));
    if (lands.size !== L) return { error: 'throws collide' };
    const sum = p.reduce((a, b) => a + b, 0);
    if (sum % L) return { error: 'not a whole number of balls' };
    return { pattern: p, n: sum / L };
  }

  // How many balls are thrown by both hands: follow each round trip through
  // the beats of a cycle (it carries as many balls as its throws add up to
  // cycles) and see whether it ever lands on the other hand's beat.
  function sharedBalls(p) {
    const L = p.length, P = L % 2 ? 2 * L : L, seen = new Set();
    let count = 0;
    for (let r = 0; r < P; r++) {
      if (p[r % L] === 0 || seen.has(r)) continue;
      let c = r, both = false, T = 0;
      do { seen.add(c); both = both || p[c % L] % 2 === 1; T += p[c % L]; c = (c + p[c % L]) % P; } while (c !== r);
      if (both) count += T / P;
    }
    return count;
  }

  function buildModel(opts) {
    const B = opts.beat;
    // A ball can't jump from the catch point to the throw point in zero
    // time, so "no dwell" means the shortest carry: a quick hairpin.
    // A 1 flies for a beat less the dwell, so a long dwell leaves it no time.
    const D = Math.max(MIN_DWELL, opts.pattern.includes(1) ? Math.min(opts.dwell, ONE_DWELL) : opts.dwell);
    // A straight-up throw comes down where it went up. A ball that only
    // ever makes even throws stays with its thrower, in a column on that
    // hand's side; a ball that goes to both hands has to come down where
    // either can reach it, in the middle, so only one ball may do that.
    // Columns look the same run backwards, so they have no reverse.
    const columns = !!opts.columns && sharedBalls(opts.pattern) <= 1;
    const reverse = !!opts.reverse && !columns;
    // Throwing outside is shown as the model run backwards in time. Run
    // backwards, a throw from beat b landing at b + k becomes a throw of the
    // same height from beat -(b + k), so build that pattern then.
    const L = opts.pattern.length;
    let pattern = opts.pattern;
    if (reverse) {
      pattern = new Array(L).fill(0);
      opts.pattern.forEach((k, i) => { if (k > 0) pattern[mod(-(i + k), L)] = k; });
    }
    const n = pattern.reduce((a, b) => a + b, 0) / L;
    const h = Math.max(...pattern);
    const geom = makeGeom(opts.spacing / 2, opts.width / 2, opts.depth / 2, BULGE, opts.reach);
    // in columns: the height of every throw, and of every catch (raised by the reach)
    const yT = 0, yC = opts.reach;

    // One flight for each throw height except 0 and 2 (no throw). On loops:
    // tangent to them, odd heights crossing to the other hand and even ones
    // coming back to the thrower, solved for the right hand (the left is its
    // mirror image). In columns: straight up and down, placed per ball below.
    const types = {};
    let err = 0, apex = -Infinity;
    for (const k of new Set(pattern)) {
      if (k === 0 || k === 2) continue;
      const cross = k % 2 === 1, F = (k - D) * B;
      if (columns) {
        const vy = (yC - yT) / F + G * F / 2;
        const v0 = { x: 0, y: vy }, v1 = { x: 0, y: vy - G * F };
        const top = yT + vy * vy / (2 * G);
        // the paddle stays flat: the ball leaves and lands square on it
        types[k] = { k, cross, F, pT: { x: 0, y: yT }, v0, v1, apex: top, sT: vy, sC: -v1.y };
        apex = Math.max(apex, top);
        continue;
      }
      const sol = solveThrow(geom, cross, F, k);
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
        sT: Math.hypot(v0.x, v0.y) * (reverse ? CATCH_GIVE : 1),
        sC: Math.hypot(v1.x, v1.y) * (reverse ? 1 : CATCH_GIVE),
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

    // Ball bookkeeping: simulate throws from beat 0, a ball thrown at beat b
    // with height k lands at beat b+k. Every ball's round trip is a multiple
    // of Q beats, so ids are periodic in Q once the first balls are in.
    const P = L % 2 === 0 ? L : 2 * L;      // beats before both hands repeat
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
    const idAt = (b) => events[mod(b, Q) + 2 * Q];

    // Columns: every ball keeps to a column of its own. A hand's own balls
    // take its columns inner to outer, in the order it first throws them,
    // and a ball the hands pass between them goes up the middle. With balls
    // on both sides the columns are spaced evenly across the whole pattern;
    // a hand juggling alone spreads them over its own loop's width. An idle
    // hand waits under the middle of the columns it uses.
    let colX = null;
    const restX = [geom.X, -geom.X];
    if (columns) {
      colX = {};
      const own = [[], []];
      for (let b = 2 * Q; b < 3 * Q; b++) {
        const id = events[b];
        if (id >= 0 && !own[b % 2].includes(id)) own[b % 2].push(id);
      }
      const shared = own[0].filter((id) => own[1].includes(id));
      const [onlyR, onlyL] = own.map((ids) => ids.filter((id) => !shared.includes(id)));
      const nR = onlyR.length, nL = onlyL.length, nS = shared.length, total = nR + nL + nS;
      if ((nR && nL) || nS) {
        const S = geom.X + geom.a - geom.a / Math.max(nR, nL, 1);   // outermost column
        const at = (i) => (total > 1 ? -S + 2 * S * i / (total - 1) : 0);
        onlyL.forEach((id, j) => { colX[id] = at(nL - 1 - j); });
        shared.forEach((id, j) => { colX[id] = at(nL + j); });
        onlyR.forEach((id, j) => { colX[id] = at(nL + nS + j); });
      } else {
        [onlyR, onlyL].forEach((ids, hi) => ids.forEach((id, j) => {
          colX[id] = (hi === 0 ? 1 : -1) * (geom.X - geom.a + 2 * geom.a * (j + 0.5) / ids.length);
        }));
      }
      own.forEach((ids, hi) => {
        if (ids.length) restX[hi] = ids.reduce((s, id) => s + colX[id], 0) / ids.length;
      });
    }

    // Each hand works through a timeline of keyframes. At each of its beats
    // it throws, or holds its ball (a 2) or is empty (a 0), and then rests at
    // the bottom of its loop. Before a beat, it catches whatever was thrown
    // to land then. Between keyframes it moves clockwise round the loop with
    // a smooth speed profile; in columns, along an arch or a dip from one
    // keyframe to the next. Columns can take longer to repeat than the
    // pattern, since the balls take turns in them.
    const landing = new Array(L).fill(0);   // height of the throw landing on each beat
    pattern.forEach((k, i) => { if (k > 0) landing[(i + k) % L] = k; });
    const Ph = columns ? lcm(P, Q) : P;
    const hands = [1, -1].map((side, hi) => {
      const kf = [], throws = new Set(), catches = new Set();
      const flights = new Map(), dots = new Map();   // for the guides
      for (let b = side === 1 ? 0 : 1; b < Ph; b += 2) {
        const k = pattern[b % L], kin = landing[b % L];
        if (types[k]) throws.add(k);
        if (types[kin]) catches.add(kin);
        if (types[kin]) {
          const ty = types[kin];
          if (columns) {
            const x = colX[idAt(b - kin)];
            kf.push({ t: (b - D) * B, p: { x, y: yC }, dir: -1, v: ty.sC, hold: true });
            dots.set(`c${x}`, { x, y: yC, kind: 'catch' });
          } else kf.push({ t: (b - D) * B, th: ty.thC, v: ty.sC, nrm: ty.nC, hold: true });
        }
        if (types[k]) {
          const ty = types[k];
          if (columns) {
            const x = colX[idAt(b)];
            kf.push({ t: b * B, p: { x, y: yT }, dir: 1, v: ty.sT, hold: false });
            flights.set(`${k}@${x}`, { ty, x });
            dots.set(`t${x}`, { x, y: yT, kind: 'throw' });
          } else kf.push({ t: b * B, th: ty.thT, v: ty.sT, nrm: ty.nT, hold: false });
        } else if (columns) {
          const x = k === 2 ? colX[idAt(b)] : restX[hi];
          kf.push({ t: b * B, p: { x, y: yT - geom.b }, dir: 0, v: 0, hold: k === 2 });
        } else {
          kf.push({ t: b * B, th: REST_TH, v: 0, nrm: Math.PI / 2, hold: k === 2 });
        }
      }
      if (!columns) {
        for (const k of throws) {
          flights.set(k, { ty: types[k] });
          dots.set(`t${k}`, { ...geom.P(types[k].thT, side), kind: 'throw' });
        }
        for (const k of catches) dots.set(`c${k}`, { ...geom.P(types[k].thC, side), kind: 'catch' });
      }
      const period = Ph * B;
      kf.forEach((f) => { f.t = mod(f.t, period); });
      kf.sort((a, b) => a.t - b.t);
      const segs = kf.map((a, i) => {
        const z = kf[(i + 1) % kf.length];
        const T = mod(z.t - a.t, period) || period;
        if (columns) {
          // Arch over after a throw or into a catch, dip under to carry; as
          // high (or low) as the loop's half-depth past the higher (lower)
          // end. A path back into its own column swings a quarter of the
          // loop's half-width: outward under a carry, inward over the top.
          const sg = a.dir || -z.dir, dy = Math.abs(z.p.y - a.p.y);
          const lift = sg * Math.sqrt(geom.b * geom.b + dy * geom.b);
          const swing = sg && Math.abs(z.p.x - a.p.x) < 1e-6 ? sg * -side * geom.a / 4 : 0;
          const path = makeArch(a.p, z.p, lift, swing);
          return {
            t0: a.t, T, len: path.len, path, hold: a.hold,
            prof: makeSegment(T, path.len, a.v, z.v),
            n0: Math.PI / 2, dn: 0,
          };
        }
        const phi0 = mod(-a.th, TAU), dphi = mod(a.th - z.th, TAU);
        const s0 = lenAt(phi0), len = lenAt(phi0 + dphi) - s0;
        const wrap = (x) => mod(x + Math.PI, TAU) - Math.PI;
        return {
          t0: a.t, T, s0, len, hold: a.hold,
          prof: makeSegment(T, len, a.v, z.v),
          n0: a.nrm, dn: wrap(z.nrm - a.nrm),
        };
      });
      return { segs, period, throws, catches, flights: [...flights.values()], dots: [...dots.values()] };
    });

    // overall size of the ball path: hand paths at the bottom, peak on top
    let xMax = 0, yMin = 0;
    if (columns) {
      for (const hand of hands) {
        for (const seg of hand.segs) {
          for (let i = 0; i <= 32; i++) {
            const p = seg.path.at(seg.len * i / 32);
            xMax = Math.max(xMax, Math.abs(p.x)); yMin = Math.min(yMin, p.y);
          }
        }
      }
    } else {
      for (let i = 0; i < 128; i++) {
        const p = geom.P(TAU * i / 128, 1);
        xMax = Math.max(xMax, p.x); yMin = Math.min(yMin, p.y);
      }
    }
    const extent = { w: 2 * xMax, h: Math.max(apex, geom.b) - yMin, xMax, yMin };

    return { n, h, pattern, geom, D, B, types, apex, err, extent, arc, hands, Q, events, L, columns, reverse, colX };
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
    if (seg.path) return seg.path.at(seg.prof(dt));
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
      if (flying && m.colX) pos.x = m.colX[id];
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
      if (!m.types[k]) continue;
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

  const STYLES = ['inside', 'outside', 'up'];   // throw style: the button cycles through them
  const opts = { pattern: patternFor(3), style: 'inside', dwell: 0.6, beat: 0.25, rate: 0.5, spacing: 0.3, width: 0.22, depth: 0.14, reach: 0.08, guides: true, trails: true, sound: false };
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
  }

  function rebuild() {
    opts.dwell = Math.min(opts.dwell, 1.9);
    // keep the two hand loops from overlapping in the middle
    const width = Math.min(opts.width, opts.spacing - 0.02);
    model = buildModel({ ...opts, width, reverse: opts.style === 'outside', columns: opts.style === 'up' });
    $('count').textContent = model.n;
    $('style').textContent = opts.style;
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
    const name = model.columns ? 'columns'
      : uniform ? (model.reverse ? 'reverse ' : '') + kind : 'siteswap' + (model.reverse ? ' (reverse)' : '');
    const peaks = Object.values(model.types).map((ty) => ty.apex * 100);
    const lo = Math.round(Math.min(...peaks)), hi = Math.round(Math.max(...peaks));
    const capped = opts.pattern.includes(1) && opts.dwell > ONE_DWELL;
    const note = opts.style === 'up' && !model.columns ? '  (straight up: one ball at most can cross)'
      : capped ? `  (dwell held to ${ONE_DWELL * 50}% for 1s)` : '';
    $('stats').textContent =
      `${name}  ${ss}${note}\n` +
      `peak ${lo === hi ? hi : lo + '–' + hi}cm\n` +
      `shape ${cm(model.extent.w)} wide × ${cm(model.extent.h)} tall`;
  }

  // Patterns offered for each number of balls, the basic one first. The ball
  // count picks the list; the menu picks from it, or "custom" opens a box
  // for any siteswap, which then sets the ball count to match. A preset can
  // also set the throw style: the pillars and columns are thrown straight up.
  const PRESETS = {
    1: [['300', 'one ball']],
    2: [['330', 'cascade'], ['4400', 'columns', 'up'], ['31', 'shower'], ['40', 'one-hand fountain'],
      ['40', 'one-hand columns', 'up'], ['420', '420'], ['312', '312']],
    3: [['3', 'cascade'], ['4440', 'pillars', 'up'], ['423', 'swap pillars', 'up'], ['51', 'shower'], ['423', '423'], ['441', '441'], ['531', '531'],
      ['4413', '4413'], ['42', 'two in one hand'], ['522', '522'], ['504', '504'], ['60', 'three in one hand'],
      ['60', 'one-hand columns', 'up']],
    4: [['4', 'fountain'], ['4', 'pillars', 'up'], ['71', 'shower'], ['53', 'half shower'], ['534', '534'],
      ['552', '552'], ['5551', '5551'], ['633', '633'], ['642', '642'], ['642', '642 in columns', 'up'], ['7531', '7531']],
    5: [['5', 'cascade'], ['663', 'swap pillars', 'up'], ['64', 'columns', 'up'], ['91', 'shower'], ['64', '64'], ['645', '645'], ['663', '663'],
      ['744', '744'], ['753', '753'], ['75751', '75751'], ['66661', '66661']],
    6: [['6', 'fountain'], ['6', 'pillars', 'up'], ['75', 'half shower'], ['855', '855']],
    7: [['7', 'cascade'], ['86', '86'], ['86', 'columns', 'up']],
    8: [['8', 'fountain'], ['8', 'pillars', 'up'], ['97', 'half shower']],
    9: [['9', 'cascade']],
  };
  const presetKey = ([value, , style]) => (style ? `${value}:${style}` : value);

  function fillPatterns() {
    const menu = $('pattern');
    const key = opts.pattern.join('') + (model.columns ? ':up' : '');
    const list = PRESETS[model.n] || [];
    menu.innerHTML = '';
    for (const p of list) menu.add(new Option(p[1] === p[0] ? p[0] : `${p[0]}  ${p[1]}`, presetKey(p)));
    menu.add(new Option('custom…', 'custom'));
    const preset = list.some((p) => presetKey(p) === key);
    menu.value = preset ? key : 'custom';
    $('customRow').hidden = preset;
    if (!preset) $('siteswap').value = opts.pattern.join('');
  }

  function setPattern(pattern) {
    opts.pattern = pattern;
    rebuild();
    fillPatterns();
  }
  // A preset with a style sets it; one without leaves the cascade styles
  // alone but brings the throws down from straight up.
  function choosePreset([value, , style]) {
    if (style) opts.style = style;
    else if (opts.style === 'up') opts.style = 'inside';
    setPattern(parseSiteswap(value).pattern);
  }
  // a new ball count starts on its basic pattern, or its pillars when throwing straight up
  const setBalls = (n) => choosePreset((opts.style === 'up' && PRESETS[n].find((p) => p[2] === 'up')) || PRESETS[n][0]);

  $('fewer').onclick = () => setBalls(Math.max(1, model.n - 1));
  $('more').onclick = () => setBalls(Math.min(9, model.n + 1));
  $('pattern').onchange = (e) => {
    if (e.target.value !== 'custom') { choosePreset(PRESETS[model.n].find((p) => presetKey(p) === e.target.value)); return; }
    $('customRow').hidden = false;
    $('siteswap').value = opts.pattern.join('');
    $('siteswap').focus();
    $('siteswap').select();
  };
  $('siteswap').onchange = () => {
    const box = $('siteswap'), r = parseSiteswap(box.value);
    box.classList.toggle('bad', !!r.error);
    $('customError').textContent = r.error || '';
    if (!r.error) setPattern(r.pattern);
  };
  $('siteswap').onkeydown = (e) => { if (e.key === 'Enter') e.target.blur(); };
  $('style').onclick = () => {
    opts.style = STYLES[(STYLES.indexOf(opts.style) + 1) % STYLES.length];
    rebuild();
    fillPatterns();
  };
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

    // full hand paths, dashed: the loops, or in columns the arches and dips
    // the hand works through over one cycle
    ctx.strokeStyle = color('--guide');
    ctx.setLineDash([4 * dpr, 5 * dpr]);
    for (const side of [1, -1]) {
      const hand = m.hands[side === 1 ? 0 : 1];
      ctx.beginPath();
      if (m.columns) {
        for (const seg of hand.segs) {
          for (let i = 0; i <= 40; i++) {
            const p = seg.path.at(seg.len * i / 40);
            i === 0 ? ctx.moveTo(sx(p.x), sy(p.y)) : ctx.lineTo(sx(p.x), sy(p.y));
          }
        }
      } else {
        for (let i = 0; i <= 160; i++) {
          const p = g.P(TAU * i / 160, side);
          i === 0 ? ctx.moveTo(sx(p.x), sy(p.y)) : ctx.lineTo(sx(p.x), sy(p.y));
        }
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
      for (const { ty, x } of m.hands[side === 1 ? 0 : 1].flights) {
        for (let i = 0; i <= 80; i++) {
          const p = flightPos(ty, ty.F * i / 80, side);
          if (x !== undefined) p.x = x;
          i === 0 ? ctx.moveTo(sx(p.x), sy(p.y)) : ctx.lineTo(sx(p.x), sy(p.y));
        }
      }
    }
    ctx.stroke();

    // throw and catch points: filled = throw, hollow = catch
    // (run backwards, the model's throws are the catches and vice versa)
    const r = 3 * dpr;
    ctx.fillStyle = color('--path');
    for (const hand of m.hands) {
      for (const d of hand.dots) {
        ctx.beginPath(); ctx.arc(sx(d.x), sy(d.y), r, 0, TAU);
        (d.kind === 'throw') !== m.reverse ? ctx.fill() : ctx.stroke();
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
      for (const b of ballsAt(model, model.reverse ? -s : s)) {
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
    for (const e of eventsBetween(model, s0, s1, model.reverse)) {
      if (e.kind === 'catch') ring(CATCH_NOTE, 0.16, 0.3, 0.5 * e.side);
    }
    for (const e of eventsBetween(model, s0 + ahead, s1 + ahead, model.reverse)) {
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

    const t = model.reverse ? -simT : simT;
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
  fillPatterns();
  requestAnimationFrame(frame);
})();
