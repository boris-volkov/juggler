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
    if (n === 1) return '300';
    if (n === 2) return '330';
    return String(n);
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
  // p1, leaving along direction d0 and arriving along d1 (unit vectors), how
  // far it runs out along them set by `lift`. Leaving up and arriving down it
  // arches over (the empty hand after a throw); leaving down and arriving up
  // it dips under (the carry). w swings the middle sideways, so a path that
  // starts and ends in the same column turns round instead of stopping dead
  // and reversing. Over f in [0, pi]: the chord eases in with (1 - cos)/2,
  // and sin(1 + cos)/2 and sin(1 - cos)/2 vanish at both ends with unit
  // slope at one end and zero at the other, so they set the end directions
  // without moving the ends. Returned as a function of distance along it,
  // for arc-length timing.
  function makeArch(p0, d0, p1, d1, lift, w) {
    const at = (f) => {
      const c = (1 - Math.cos(f)) / 2, s = Math.sin(f);
      const g0 = lift * s * (1 + Math.cos(f)) / 2, g1 = lift * s * (1 - Math.cos(f)) / 2;
      return {
        x: p0.x + (p1.x - p0.x) * c + d0.x * g0 - d1.x * g1 + w * s * s,
        y: p0.y + (p1.y - p0.y) * c + d0.y * g0 - d1.y * g1,
      };
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

  // A pattern is a schedule of throws over a cycle of P beats:
  // table[b] = [right, left], each { k, to } for a hand that throws (or
  // holds, or waits empty) on beat b, with to = 0 for the right hand and 1
  // for the left; or null for a hand that sits the beat out. A ball thrown
  // with height k lands k beats later in hand `to`.
  //   async, e.g. 534: the hands take turns, and odd throws cross;
  //   sync, e.g. (4x,0)(4,4)*: both hands together every other beat, x
  //   crosses, and * repeats it all with the hands swapped.
  // It is valid when no two balls land in the same hand at once and every
  // ball lands on a beat its hand throws, which makes the average a whole
  // number: the number of balls.
  function parseSiteswap(text) {
    const s = text.replace(/\s+/g, '').toLowerCase();
    const table = [];
    const sync = s.startsWith('(');
    if (/^[0-9]+$/.test(s)) {
      const p = [...s].map(Number), L = p.length;
      for (let b = 0; b < (L % 2 ? 2 * L : L); b++) {
        const k = p[b % L], row = [null, null];
        row[b % 2] = { k, to: (b + k) % 2 };
        table.push(row);
      }
    } else if (/^(\(\dx?,\dx?\))+\*?$/.test(s)) {
      let pairs = [...s.matchAll(/\((\d)(x?),(\d)(x?)\)/g)].map((m) => [[+m[1], !!m[2]], [+m[3], !!m[4]]]);
      if (s.endsWith('*')) pairs = pairs.concat(pairs.map(([r, l]) => [l, r]));
      if (pairs.some((pr) => pr.some(([k]) => k % 2))) return { error: 'sync throws are even' };
      for (const [r, l] of pairs) {
        table.push([{ k: r[0], to: r[1] ? 1 : 0 }, { k: l[0], to: l[1] ? 0 : 1 }]);
        table.push([null, null]);
      }
    } else return { error: 'digits, or sync like (4,4)' };
    const P = table.length, lands = new Set();
    let sum = 0, h = 0, count = 0;
    table.forEach((row, b) => row.forEach((e) => {
      if (!e || e.k === 0) return;
      count++; sum += e.k; h = Math.max(h, e.k);
      const at = table[(b + e.k) % P][e.to];
      if (at && at.k > 0) lands.add(`${(b + e.k) % P},${e.to}`);
    }));
    if (h < 3) return { error: 'needs a throw of 3 or more' };
    if (lands.size !== count) return { error: 'throws collide' };
    return { text: s, table, P, n: sum / P, h, sync };
  }

  const unit = (x, y) => { const l = Math.hypot(x, y); return { x: x / l, y: y / l }; };

  // the kind of flight a throw makes, or null for no flight: an empty hand
  // (0) or a ball held a beat (a 2 back to the same hand)
  const throwKey = (e, hi) =>
    (!e || e.k === 0 || (e.k === 2 && e.to === hi) ? null : `${e.k}${e.to !== hi ? 'x' : ''}`);

  // Follow every ball's round trip through a cycle of the table; calls
  // visit(beats, hands) once per round trip with the beats and hands it
  // throws from. Each round trip carries (its throws added up) / P balls.
  function roundTrips(table, visit) {
    const P = table.length, seen = new Set();
    table.forEach((row, b0) => row.forEach((e0, h0) => {
      if (!e0 || e0.k === 0 || seen.has(`${b0},${h0}`)) return;
      const trip = [];
      let b = b0, hi = h0;
      do {
        seen.add(`${b},${hi}`);
        const e = table[b][hi];
        trip.push({ b, hi, e });
        b = (b + e.k) % P; hi = e.to;
      } while (b !== b0 || hi !== h0);
      visit(trip);
    }));
  }

  // how many balls are thrown by both hands
  function sharedBalls(table) {
    let count = 0;
    roundTrips(table, (trip) => {
      if (trip.some(({ hi, e }) => e.to !== hi)) count += trip.reduce((s, { e }) => s + e.k, 0) / table.length;
    });
    return count;
  }

  function buildModel(opts) {
    const B = opts.beat;
    const pat = parseSiteswap(opts.siteswap);
    const ones = pat.table.some((row) => row.some((e) => e && e.k === 1));
    // A ball can't jump from the catch point to the throw point in zero
    // time, so "no dwell" means the shortest carry: a quick hairpin.
    // A 1 flies for a beat less the dwell, so a long dwell leaves it no time.
    const D = Math.max(MIN_DWELL, ones ? Math.min(opts.dwell, ONE_DWELL) : opts.dwell);
    // A straight-up throw comes down where it went up. A ball that never
    // changes hands stays in a column on its hand's side; a ball that goes
    // to both hands has to come down where either can reach it, in the
    // middle, so only one ball may do that.
    // Or, as a rainbow, it is thrown from the outside of one hand in an arc
    // over everything to the outside of the other.
    // Columns look the same run backwards, so they have no reverse.
    const shared = sharedBalls(pat.table);
    const columns = !!opts.columns && shared <= 1;
    const rainbow = columns && !!opts.rainbow && shared === 1;
    const reverse = !!opts.reverse && !columns;
    // Throwing outside is shown as the model run backwards in time. Run
    // backwards, a throw from beat b landing at b + k becomes a throw of the
    // same height from beat -(b + k), from the catching hand to the
    // throwing one, so build that pattern then.
    const P = pat.P, n = pat.n, h = pat.h;
    let table = pat.table;
    if (reverse) {
      table = pat.table.map((row) => row.map((e, hi) => (e ? { k: 0, to: hi } : null)));
      pat.table.forEach((row, b) => row.forEach((e, hi) => {
        if (e && e.k > 0) table[mod(-(b + e.k), P)][e.to] = { k: e.k, to: hi };
      }));
    }
    const geom = makeGeom(opts.spacing / 2, opts.width / 2, opts.depth / 2, BULGE, opts.reach);
    // in columns: the height of every throw, and of every catch (raised by
    // the reach)
    const yT = 0, yC = opts.reach;
    // Under a rainbow the other balls stand in pillars close together in the
    // middle, three ball radii apart, and the rainbow leaves and lands well
    // outside them: at the outer edge of each hand's loop, and at least
    // five times as far out as the outermost pillar. Nearer, it can reach a
    // pillar just as that pillar's ball comes down through the same spot.
    let pillars = 0;
    if (rainbow) roundTrips(table, (trip) => {
      if (trip.every(({ hi, e }) => e.to === hi)) pillars += trip.reduce((s, { e }) => s + e.k, 0) / P;
    });
    const pillarX = Math.max(0, pillars - 1) * 1.5 * BALL_R;
    const rainX = Math.max(geom.X + geom.a, 5 * pillarX);

    // One flight for each kind of throw: its height, and whether it crosses
    // to the other hand. On loops: tangent to them, solved for the right
    // hand (the left is its mirror image). In columns: straight up and
    // down, placed per ball below; a rainbow arcs from rainX on the
    // thrower's side to rainX on the other.
    const types = {};
    let err = 0, apex = -Infinity;
    for (const row of table) {
      row.forEach((e, hi) => {
        const key = throwKey(e, hi);
        if (!key || types[key]) return;
        const k = e.k, cross = e.to !== hi, F = (k - D) * B;
        if (columns) {
          const arc = rainbow && cross;
          const vy = (yC - yT) / F + G * F / 2, vx = arc ? -2 * rainX / F : 0;
          const v0 = { x: vx, y: vy }, v1 = { x: vx, y: vy - G * F };
          // arrival at the right hand is the mirror image when the throw crosses
          const vc = cross ? { x: -v1.x, y: v1.y } : v1;
          const top = yT + vy * vy / (2 * G);
          // the paddle faces along the launch and back up the arrival: flat
          // under a straight-up throw, tipped in under a rainbow
          types[key] = {
            k, cross, F, arc, pT: { x: arc ? rainX : 0, y: yT }, y0: yT, v0, v1, vc, apex: top,
            nT: Math.atan2(v0.y, v0.x), nC: Math.atan2(-vc.y, -vc.x),
            sT: Math.hypot(v0.x, v0.y), sC: Math.hypot(v1.x, v1.y),
          };
          apex = Math.max(apex, top);
          return;
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
        types[key] = {
          k, cross, F, thT: sol.thT, thC: sol.thC, pT, v0, v1, apex: top,
          // where the ball really leaves the hand (run backwards, the catch point)
          y0: reverse ? pC.y : pT.y,
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
      });
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
    // events[b] = [right, left]: the id of the ball each hand throws, or -1.
    let Q = 1;
    roundTrips(table, (trip) => { Q = lcm(Q, trip.reduce((s, { e }) => s + e.k, 0)); });
    const lands = new Map();
    const events = [];
    let nextId = 0;
    for (let b = 0; b < 4 * Q; b++) {
      events.push(table[b % P].map((e, hi) => {
        if (!e || e.k === 0) return -1;
        let id = lands.get(`${b},${hi}`);
        if (id === undefined) id = nextId++;
        lands.set(`${b + e.k},${e.to}`, id);
        return id;
      }));
    }
    const idAt = (b, hi) => events[mod(b, Q) + 2 * Q][hi];

    // Columns: every ball keeps to a column of its own. A hand's own balls
    // take its columns inner to outer, in the order it first throws them,
    // and a ball the hands pass between them goes up the middle (or, as a
    // rainbow, leaves and lands at rainX on each side: colX holds [right,
    // left] for it). With balls on both sides the columns are spaced evenly
    // across the whole pattern; a hand juggling alone spreads them over its
    // own loop's width. An idle hand waits under the middle of the columns
    // it uses.
    let colX = null;
    const colAt = (id, hi) => (Array.isArray(colX[id]) ? colX[id][hi] : colX[id]);
    const restX = [geom.X, -geom.X];
    if (columns) {
      colX = {};
      const own = [[], []];
      for (let b = 2 * Q; b < 3 * Q; b++) {
        events[b].forEach((id, hi) => { if (id >= 0 && !own[hi].includes(id)) own[hi].push(id); });
      }
      const shared = own[0].filter((id) => own[1].includes(id));
      const [onlyR, onlyL] = own.map((ids) => ids.filter((id) => !shared.includes(id)));
      const mid = rainbow ? [] : shared;
      const nR = onlyR.length, nL = onlyL.length, nS = mid.length, total = nR + nL + nS;
      if ((nR && nL) || nS) {
        const S = rainbow ? pillarX : geom.X + geom.a - geom.a / Math.max(nR, nL, 1);   // outermost column
        const at = (i) => (total > 1 ? -S + 2 * S * i / (total - 1) : 0);
        onlyL.forEach((id, j) => { colX[id] = at(nL - 1 - j); });
        mid.forEach((id, j) => { colX[id] = at(nL + j); });
        onlyR.forEach((id, j) => { colX[id] = at(nL + nS + j); });
      } else {
        [onlyR, onlyL].forEach((ids, hi) => ids.forEach((id, j) => {
          colX[id] = (hi === 0 ? 1 : -1) * (geom.X - geom.a + 2 * geom.a * (j + 0.5) / ids.length);
        }));
      }
      if (rainbow) shared.forEach((id) => { colX[id] = [rainX, -rainX]; });
      own.forEach((ids, hi) => {
        if (ids.length) restX[hi] = ids.reduce((s, id) => s + colAt(id, hi), 0) / ids.length;
      });
    }

    // Each hand works through a timeline of keyframes. On each of its beats
    // it throws, or holds its ball (a 2) or is empty (a 0), and then rests at
    // the bottom of its loop. Before a beat, it catches whatever was thrown
    // to land then. Between keyframes it moves clockwise round the loop with
    // a smooth speed profile; in columns, along an arch or a dip from one
    // keyframe to the next. Columns can take longer to repeat than the
    // pattern, since the balls take turns in them.
    const landing = table.map(() => [null, null]);   // the throw landing in each hand on each beat
    table.forEach((row, b) => row.forEach((e, hi) => {
      if (e && e.k > 0) landing[(b + e.k) % P][e.to] = { k: e.k, from: hi, key: throwKey(e, hi) };
    }));
    const Ph = columns ? lcm(P, Q) : P;
    const hands = [1, -1].map((side, hi) => {
      const kf = [], throws = new Set(), catches = new Set();
      const flights = new Map(), dots = new Map();   // for the guides
      const busy = table.some((row) => row[hi] && row[hi].k > 0);   // this hand juggles at all
      for (let b = 0; b < Ph; b++) {
        const e = table[b % P][hi];
        if (!e) continue;
        const lin = landing[b % P][hi];
        const tyIn = lin && lin.key && types[lin.key];
        if (tyIn) {
          catches.add(lin.key);
          if (columns) {
            const x = colAt(idAt(b - lin.k, lin.from), hi);
            kf.push({ t: (b - D) * B, p: { x, y: yC }, d: unit(side * tyIn.vc.x, tyIn.vc.y),
              v: tyIn.sC, nrm: tyIn.nC, hold: true });
            dots.set(`c${x}`, { x, y: yC, kind: 'catch' });
          } else kf.push({ t: (b - D) * B, th: tyIn.thC, v: tyIn.sC, nrm: tyIn.nC, hold: true });
        }
        const key = throwKey(e, hi), ty = key && types[key];
        if (ty) {
          throws.add(key);
          if (columns) {
            const x = colAt(idAt(b, hi), hi);
            kf.push({ t: b * B, p: { x, y: yT }, d: unit(side * ty.v0.x, ty.v0.y), v: ty.sT, nrm: ty.nT, hold: false });
            flights.set(`${key}@${x}`, { ty, x: ty.arc ? undefined : x });
            dots.set(`t${x}`, { x, y: yT, kind: 'throw' });
          } else kf.push({ t: b * B, th: ty.thT, v: ty.sT, nrm: ty.nT, hold: false });
        } else if (columns) {
          // An empty hand that has juggling to get back to goes straight on
          // from its last throw to its next catch, rather than dropping to a
          // rest on the way and coming back up.
          if (e.k === 0 && busy) continue;
          const x = e.k === 2 ? colAt(idAt(b, hi), hi) : restX[hi];
          kf.push({ t: b * B, p: { x, y: yT - geom.b }, d: null, v: 0, nrm: Math.PI / 2, hold: e.k === 2 });
        } else {
          kf.push({ t: b * B, th: REST_TH, v: 0, nrm: Math.PI / 2, hold: e.k === 2 });
        }
      }
      if (!columns) {
        for (const key of throws) {
          flights.set(key, { ty: types[key] });
          dots.set(`t${key}`, { ...geom.P(types[key].thT, side), kind: 'throw' });
        }
        for (const key of catches) dots.set(`c${key}`, { ...geom.P(types[key].thC, side), kind: 'catch' });
      }
      const period = Ph * B;
      kf.forEach((f) => { f.t = mod(f.t, period); });
      kf.sort((a, b) => a.t - b.t);
      const wrap = (x) => mod(x + Math.PI, TAU) - Math.PI;
      const segs = kf.map((a, i) => {
        const z = kf[(i + 1) % kf.length];
        const T = mod(z.t - a.t, period) || period;
        if (columns) {
          // Arch over after a throw or into a catch, dip under to carry; as
          // high (or low) as the loop's half-depth past the higher (lower)
          // end. A resting end leaves (or arrives) vertically, opposite to
          // the other end. A path back into its own column swings a quarter
          // of the loop's half-width: outward under a carry, inward over the top.
          const sg = a.d ? Math.sign(a.d.y) : z.d ? -Math.sign(z.d.y) : 0;
          const d0 = a.d || { x: 0, y: sg }, d1 = z.d || { x: 0, y: -sg };
          const dy = Math.abs(z.p.y - a.p.y);
          const lift = sg ? Math.sqrt(geom.b * geom.b + dy * geom.b) : 0;
          const swing = sg && Math.abs(z.p.x - a.p.x) < 1e-6 ? sg * -side * geom.a / 4 : 0;
          const path = makeArch(a.p, d0, z.p, d1, lift, swing);
          return {
            t0: a.t, T, len: path.len, path, hold: a.hold,
            prof: makeSegment(T, path.len, a.v, z.v),
            n0: a.nrm, dn: wrap(z.nrm - a.nrm),
          };
        }
        const phi0 = mod(-a.th, TAU), dphi = mod(a.th - z.th, TAU);
        const s0 = lenAt(phi0), len = lenAt(phi0 + dphi) - s0;
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

    // The heights a pattern mixes, tallest first: each throw's rise above the
    // point it leaves the hand, as a ratio of the lowest. A 1 is a flat pass
    // rather than a height, so it isn't counted; a 4 and a 4x are one height.
    const rises = new Map();
    for (const ty of Object.values(types)) {
      const rise = ty.apex - ty.y0, seen = rises.get(ty.k);
      if (ty.k !== 1 && (!seen || rise > seen.rise)) rises.set(ty.k, { k: ty.k, apex: ty.apex, rise });
    }
    let heights = [...rises.values()].sort((a, b) => b.rise - a.rise);
    const lowest = heights.length ? heights[heights.length - 1].rise : 0;
    if (heights.length < 2 || lowest < 1e-3) heights = [];
    for (const row of heights) row.ratio = row.rise / lowest;

    return {
      n, h, P, table, sync: pat.sync, ones, geom, D, B, types, apex, err, extent, arc, hands, Q, events,
      columns, rainbow, reverse, colX, colAt, heights,
    };
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
      m.table[b % m.P].forEach((e, hi) => {
        const id = m.events[b][hi];
        if (id < 0 || tb >= b + e.k) return;
        const tau = (tb - b) * m.B;
        // a held 2 never leaves the hand
        const ty = m.types[throwKey(e, hi)];
        const flying = ty && tau < ty.F;
        const pos = flying ? flightPos(ty, tau, hi === 0 ? 1 : -1) : handPos(m, t, e.to === 0 ? 1 : -1);
        if (flying && m.colX && !ty.arc) pos.x = m.colAt(id, hi);
        out.push({ id, ...pos, held: !flying });
      });
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
      m.table[mod(b, m.P)].forEach((e, hi) => {
        if (!m.types[throwKey(e, hi)]) return;
        const id = m.events[mod(b, m.Q) + 2 * m.Q][hi];
        add(b * m.B, 'throw', id, hi === 0 ? 1 : -1);
        add((b + e.k - m.D) * m.B, 'catch', id, e.to === 0 ? 1 : -1);
      });
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

  const STYLES = ['inside', 'outside', 'up', 'rainbow'];   // throw style: the button cycles through them
  const straightUp = (style) => style === 'up' || style === 'rainbow';
  const opts = { siteswap: patternFor(3), style: 'inside', dwell: 0.6, beat: 0.25, rate: 0.5, spacing: 0.3, width: 0.22, depth: 0.14, reach: 0.08, guides: true, heights: false, trail: 0.35, glow: false, sound: false };
  let model = buildModel(opts);
  let simT = 0, paused = false, last = performance.now();
  let view = null;

  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  // seventies enamel: burnt orange, teal, mustard, olive, plum, navy, rust, sea green, sienna
  // Glow mode: a light show on a night stage, in neon
  const neon = ['#ff3d6e', '#2ef0ff', '#ffd43a', '#78ff5a', '#b66cff', '#ff8b24', '#4a83ff', '#ff5ad6', '#5affc6'];
  const NIGHT = { '--bg': '#060509', '--guide': 'rgba(255, 255, 255, 0.07)',
    '--path': 'rgba(255, 255, 255, 0.14)', '--ink': 'rgba(255, 255, 255, 0.35)' };
  // stage colours: the theme's, or the night stage's while glowing
  const stage = (name) => (opts.glow ? NIGHT[name] : color(name));
  const ballColor = (id) => (opts.glow ? neon : palette)[id % palette.length];

  // Theme: beige or dark, remembered between visits where the browser allows
  let theme = 'beige';
  try { if (localStorage.getItem('juggler-theme') === 'dark') theme = 'dark'; } catch (e) { /* no storage */ }
  function setTheme(t) {
    theme = t;
    document.documentElement.dataset.theme = t;
    $('theme').textContent = t;
    try { localStorage.setItem('juggler-theme', t); } catch (e) { /* no storage */ }
  }
  setTheme(theme);

  const showTrail = () => { $('trailVal').textContent = opts.trail ? opts.trail.toFixed(2) + 'b' : 'off'; };

  const palette = ['#c4552f', '#2f7275', '#d9a12b', '#6e8b3d', '#8c3b5a', '#34477a', '#de7a3e', '#4f9a8c', '#9a5a32'];

  // Browsers restore form controls on a reload but the script starts from
  // its defaults, so set the controls from the defaults to keep them agreeing.
  function syncControls() {
    const values = { dwell: opts.dwell, spacing: opts.spacing, width: opts.width,
      depth: opts.depth, rise: opts.reach, beat: opts.beat, rate: opts.rate, trail: opts.trail };
    for (const [id, v] of Object.entries(values)) $(id).value = v;
    for (const id of ['guides', 'heights', 'glow', 'sound']) $(id).checked = opts[id];
  }

  function rebuild() {
    opts.dwell = Math.min(opts.dwell, 1.9);
    // keep the two hand loops from overlapping in the middle
    const width = Math.min(opts.width, opts.spacing - 0.02);
    model = buildModel({ ...opts, width, reverse: opts.style === 'outside',
      columns: straightUp(opts.style), rainbow: opts.style === 'rainbow' });
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
    const ss = opts.siteswap;
    const uniform = /^(\d)\1*$/.test(ss) || ss === '300' || ss === '330';
    const kind = Object.values(model.types).every((ty) => ty.cross) ? 'cascade' : 'fountain';
    const name = model.rainbow ? 'rainbow' : model.columns ? 'columns'
      : uniform ? (model.reverse ? 'reverse ' : '') + kind
      : (model.sync ? 'sync' : 'siteswap') + (model.reverse ? ' (reverse)' : '');
    const peaks = Object.values(model.types).map((ty) => ty.apex * 100);
    const lo = Math.round(Math.min(...peaks)), hi = Math.round(Math.max(...peaks));
    const capped = model.ones && opts.dwell > ONE_DWELL;
    const note = straightUp(opts.style) && !model.columns ? '  (straight up: one ball at most can cross)'
      : opts.style === 'rainbow' && !model.rainbow ? '  (rainbow: needs one ball to cross)'
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
  // Swap pillars are synchronous, so the middle ball can change hands and
  // still fly as high as the others.
  const PRESETS = {
    1: [['300', 'one ball']],
    2: [['330', 'cascade'], ['4400', 'columns', 'up'], ['31', 'shower'], ['40', 'one-hand fountain'],
      ['40', 'one-hand columns', 'up'], ['420', '420'], ['312', '312']],
    3: [['3', 'cascade'], ['4440', 'pillars', 'up'], ['(4x,0)(4,4)*', 'swap pillars', 'up'],
      ['(4x,0)(4,4)*', 'rainbow over pillars', 'rainbow'], ['51', 'shower'], ['423', '423'], ['441', '441'], ['531', '531'],
      ['4413', '4413'], ['42', 'two in one hand'], ['522', '522'], ['504', '504'], ['60', 'three in one hand'],
      ['60', 'one-hand columns', 'up']],
    4: [['4', 'fountain'], ['4', 'pillars', 'up'], ['71', 'shower'], ['53', 'half shower'], ['534', '534'],
      ['552', '552'], ['5551', '5551'], ['633', '633'], ['642', '642'], ['642', '642 in columns', 'up'], ['7531', '7531'],
      ['(4,4)', 'sync fountain']],
    5: [['5', 'cascade'], ['(6x,0)(6,6)(6,6)*', 'swap pillars', 'up'],
      ['(6x,0)(6,6)(6,6)*', 'rainbow over pillars', 'rainbow'], ['64', 'columns', 'up'], ['91', 'shower'], ['64', '64'], ['645', '645'], ['663', '663'],
      ['744', '744'], ['753', '753'], ['75751', '75751'], ['66661', '66661'], ['(6x,4)*', '(6x,4)*']],
    6: [['6', 'fountain'], ['6', 'pillars', 'up'], ['75', 'half shower'], ['855', '855'], ['(6,6)', 'sync fountain']],
    7: [['7', 'cascade'], ['86', '86'], ['86', 'columns', 'up']],
    8: [['8', 'fountain'], ['8', 'pillars', 'up'], ['97', 'half shower']],
    9: [['9', 'cascade']],
  };
  const presetKey = ([value, , style]) => (style ? `${value}:${style}` : value);

  function fillPatterns() {
    const menu = $('pattern');
    const key = opts.siteswap + (model.rainbow ? ':rainbow' : model.columns ? ':up' : '');
    const list = PRESETS[model.n] || [];
    menu.innerHTML = '';
    // sync notation is long, so those show by name alone (the stats show the notation)
    for (const p of list) {
      menu.add(new Option(p[1] === p[0] || p[0].startsWith('(') ? p[1] : `${p[0]}  ${p[1]}`, presetKey(p)));
    }
    menu.add(new Option('custom…', 'custom'));
    const preset = list.some((p) => presetKey(p) === key);
    menu.value = preset ? key : 'custom';
    $('customRow').hidden = preset;
    if (!preset) $('siteswap').value = opts.siteswap;
  }

  function setPattern(siteswap) {
    opts.siteswap = siteswap;
    rebuild();
    fillPatterns();
  }
  // A preset with a style sets it; one without leaves the cascade styles
  // alone but brings the throws down from straight up.
  function choosePreset([value, , style]) {
    if (style) opts.style = style;
    else if (straightUp(opts.style)) opts.style = 'inside';
    setPattern(value);
  }
  // A new ball count starts on its basic pattern, or when throwing straight
  // up on its first pattern in the same style (its pillars or its rainbow).
  const setBalls = (n) => choosePreset(
    (opts.style === 'rainbow' && PRESETS[n].find((p) => p[2] === 'rainbow')) ||
    (straightUp(opts.style) && PRESETS[n].find((p) => p[2] === 'up')) || PRESETS[n][0]);

  $('fewer').onclick = () => setBalls(Math.max(1, model.n - 1));
  $('more').onclick = () => setBalls(Math.min(9, model.n + 1));
  $('pattern').onchange = (e) => {
    if (e.target.value !== 'custom') { choosePreset(PRESETS[model.n].find((p) => presetKey(p) === e.target.value)); return; }
    $('customRow').hidden = false;
    $('siteswap').value = opts.siteswap;
    $('siteswap').focus();
    $('siteswap').select();
  };
  $('siteswap').onchange = () => {
    const box = $('siteswap'), r = parseSiteswap(box.value);
    box.classList.toggle('bad', !!r.error);
    $('customError').textContent = r.error || '';
    if (!r.error) setPattern(r.text);
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
  $('trail').oninput = (e) => { opts.trail = +e.target.value; showTrail(); };
  $('heights').onchange = (e) => { opts.heights = e.target.checked; };
  $('glow').onchange = (e) => { opts.glow = e.target.checked; };
  $('theme').onclick = () => setTheme(theme === 'dark' ? 'beige' : 'dark');
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

  const RULER_W = 80;   // room for the height ruler beside the pattern (css pixels)
  const showHeights = () => opts.heights && model.heights.length > 0;

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
    const ruler = showHeights() ? 2 * RULER_W * dpr : 0;   // kept even, so the pattern stays centred
    const fit = (x0, y0) => {
      const w = canvas.width - x0, h = canvas.height - y0;
      const s = Math.min((w - ruler) * (1 - 2 * pad) / (2 * xMax), h * (1 - 2 * pad) / (yMax - yMin));
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
    ctx.strokeStyle = stage('--guide');
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
    ctx.strokeStyle = stage('--path');
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
    ctx.fillStyle = stage('--path');
    for (const hand of m.hands) {
      for (const d of hand.dots) {
        ctx.beginPath(); ctx.arc(sx(d.x), sy(d.y), r, 0, TAU);
        (d.kind === 'throw') !== m.reverse ? ctx.fill() : ctx.stroke();
      }
    }
  }

  // A ruler beside the pattern: a faint level across it at each throw's peak,
  // labelled with the throw and its height as a ratio of the lowest.
  function drawHeights() {
    const dpr = devicePixelRatio || 1, hs = model.heights;
    const x0 = sx(-model.extent.xMax), x = sx(model.extent.xMax + BALL_R + 0.06);
    ctx.lineWidth = dpr;
    ctx.strokeStyle = stage('--path');
    ctx.beginPath();
    ctx.moveTo(x, sy(0));
    ctx.lineTo(x, sy(hs[0].apex));
    ctx.stroke();
    ctx.fillStyle = stage('--ink');
    ctx.font = `${12 * dpr}px "IBM Plex Mono", ui-monospace, Consolas, monospace`;
    ctx.textBaseline = 'middle';
    let last = -Infinity;
    for (const { k, apex, ratio } of hs) {
      const y = sy(apex);
      ctx.setLineDash([2 * dpr, 4 * dpr]);
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 6 * dpr, y);
      ctx.stroke();
      last = Math.max(y, last + 15 * dpr);   // labels of near heights stay apart
      ctx.fillText(`${k}  ${ratio < 1.05 ? 1 : ratio.toFixed(1)}×`, x + 10 * dpr, last);
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
    ctx.strokeStyle = stage('--ink');
    ctx.lineWidth = thick * view.s;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(sx(cx - ux * half), sy(cy - uy * half));
    ctx.lineTo(sx(cx + ux * half), sy(cy + uy * half));
    ctx.stroke();
  }

  // The motion is a closed-form function of time, so a trail is just the
  // same balls sampled a little in the past; no history buffer needed. It is
  // opts.trail beats long, sampled finely enough for the fast swing through
  // the bottom, and drawn in short runs that fade toward the tail. Glowing,
  // the runs add up as light: a faint wide halo, the coloured band, and a
  // white-hot core, like a glowstick swung in the dark.
  function drawTrails() {
    const span = opts.trail * model.B;
    const steps = Math.min(1200, Math.max(60, Math.ceil(120 * opts.trail / 0.35)));
    const trails = new Map();
    for (let i = steps; i >= 0; i--) {
      const s = simT - span * i / steps;
      for (const b of ballsAt(model, model.reverse ? -s : s)) {
        if (!trails.has(b.id)) trails.set(b.id, []);
        trails.get(b.id).push(b);
      }
    }
    const w = 2 * BALL_R * view.s;
    ctx.lineCap = 'butt';   // round caps overlap and look beaded at low alpha
    ctx.lineJoin = 'round';
    if (!opts.glow) {
      const RUN = 6;        // samples per run
      for (const [id, pts] of trails) {
        ctx.strokeStyle = ballColor(id);
        ctx.lineWidth = w;
        for (let i = 0; i < pts.length - 1; i += RUN) {
          const j = Math.min(pts.length - 1, i + RUN);
          const f = (i + j) / 2 / (pts.length - 1);   // 0 oldest .. 1 newest
          ctx.globalAlpha = 0.3 * f * f;
          ctx.beginPath();
          ctx.moveTo(sx(pts[i].x), sy(pts[i].y));
          for (let k = i + 1; k <= j; k++) ctx.lineTo(sx(pts[k].x), sy(pts[k].y));
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      return;
    }
    // Light adds up, so a joint between runs would show as a seam. Instead
    // each pass is a stack of unbroken strokes, every one from some point to
    // the head: the newer the stretch, the more of them light it, which
    // fades the tail with no joints along the trail. Each stroke is whole, so
    // round caps don't overlap themselves, and the head, where every stroke
    // ends, comes out rounded like the ball's glow instead of cut square.
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    const LAYERS = 40;
    const passes = [[2.4 * w, 0.12], [w, 0.75], [0.3 * w, 0.6, '#fff']];   // width, alpha at the head, tint
    for (const [id, all] of trails) {
      const stride = Math.max(1, Math.floor(all.length / 300));
      const pts = all.filter((p, i) => i % stride === 0 || i === all.length - 1);
      for (const [width, alpha, tint] of passes) {
        ctx.strokeStyle = tint || ballColor(id);
        ctx.lineWidth = width;
        ctx.globalAlpha = alpha / LAYERS;
        for (let l = 0; l < LAYERS; l++) {
          const i0 = Math.floor((pts.length - 1) * l / LAYERS);
          ctx.beginPath();
          ctx.moveTo(sx(pts[i0].x), sy(pts[i0].y));
          for (let k = i0 + 1; k < pts.length; k++) ctx.lineTo(sx(pts[k].x), sy(pts[k].y));
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }

  // A glowing ball: white-hot in the middle, its colour at the rim, and a
  // halo fading out to four times its size.
  function drawBall(ball) {
    const x = sx(ball.x), y = sy(ball.y), r = BALL_R * view.s, col = ballColor(ball.id);
    if (!opts.glow) {
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TAU);
      ctx.fill();
      return;
    }
    const g = ctx.createRadialGradient(x, y, 0, x, y, 4 * r);
    g.addColorStop(0, '#fff');
    g.addColorStop(0.12, '#fff');
    g.addColorStop(0.25, col);
    g.addColorStop(0.4, col + '66');
    g.addColorStop(1, col + '00');
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, 4 * r, 0, TAU);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
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

    ctx.fillStyle = stage('--bg');
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (opts.guides) drawGuides();
    if (showHeights()) drawHeights();

    const t = model.reverse ? -simT : simT;
    drawPaddle(t, 1);
    drawPaddle(t, -1);
    if (opts.trail > 0) drawTrails();
    for (const ball of ballsAt(model, t)) drawBall(ball);
    requestAnimationFrame(frame);
  }

  syncControls();
  showTrail();
  rebuild();
  fillPatterns();
  requestAnimationFrame(frame);
})();
