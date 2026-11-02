# juggler

A juggler reduced to its mathematics: balls under real gravity, and two paddles for hands.

Each hand travels a loop, and every flight is a gravity parabola that leaves one loop tangentially and lands tangentially on the next, with the hand matching the ball's speed at both ends. In a cascade the two flights and the two carries close into a figure-8.

Run it: open `index.html`, or https://boris-volkov.github.io/juggler/

- 1–9 balls (odd counts cascade, even counts fountain; 1 and 2 use `300` and `330`)
- throw outside (reverse cascade) or inside (standard) — the one is exactly the other run backwards in time
- or straight up, for pillars and columns: each ball keeps to a column of its own, and one ball may be passed between the hands up the middle (swap pillars, `(4x,0)(4,4)*`)
- or as a rainbow: the same, but the ball that changes hands arcs from the outside of one hand, over pillars standing close in the middle, to the outside of the other
- any async siteswap (`534`, `441`, `51`) or synchronous one (`(4,4)`, `(6x,4)*`), from the pattern menu or typed in
- dwell, tempo, playback speed, and the spacing, width and depth of the hand loops

Keys: space pauses, 1–9 sets the balls, ← → step a frame.
