// Scroll-driven step tracking.
//
// Observes .step elements and reports the index of the one currently holding
// the reader's attention. Uses a viewport band rather than a single threshold
// so the active step changes at a predictable point regardless of step height,
// and falls back to "all steps visible" behaviour when the observer is
// unavailable or the reader prefers reduced motion.

export function watchSteps(root, onChange, {rail} = {}) {
  const steps = Array.from(root.querySelectorAll(".step"));
  if (!steps.length) return () => {};

  // Progress indicator: built from the steps themselves so the count cannot
  // drift out of step with the prose. Dots are buttons, so the walkthrough can
  // be navigated without scrolling.
  let dots = [];
  let counter = null;
  if (rail) {
    counter = document.createElement("span");
    counter.className = "scrolly-rail-count";

    const track = document.createElement("div");
    track.className = "scrolly-rail-track";
    dots = steps.map((_, i) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = "scrolly-dot";
      dot.setAttribute("aria-label", `Go to step ${i + 1} of ${steps.length}`);
      dot.addEventListener("click", () => {
        setActive(i);
        steps[i].scrollIntoView({behavior: "smooth", block: "center"});
      });
      track.append(dot);
      return dot;
    });

    rail.replaceChildren(counter, track);
  }

  let active = -1;
  const setActive = (i) => {
    if (i === active) return;
    active = i;
    steps.forEach((el, j) => el.classList.toggle("is-active", j === i));
    dots.forEach((dot, j) => {
      dot.classList.toggle("is-active", j === i);
      dot.classList.toggle("is-done", j < i);
      dot.setAttribute("aria-current", j === i ? "step" : "false");
    });
    if (counter) counter.textContent = `Step ${i + 1} of ${steps.length}`;
    onChange(i);
  };

  if (typeof IntersectionObserver !== "function") {
    steps.forEach((el) => el.classList.add("is-active"));
    onChange(steps.length - 1);
    return () => {};
  }

  // Band across the middle of the viewport. The inset lives in one place so the
  // observer's margin and the measurement below cannot drift apart.
  const BAND_INSET_PCT = 40;
  const bandCoverage = (el) => {
    const top = window.innerHeight * (BAND_INSET_PCT / 100);
    const bottom = window.innerHeight * (1 - BAND_INSET_PCT / 100);
    const rect = el.getBoundingClientRect();
    return Math.min(rect.bottom, bottom) - Math.max(rect.top, top);
  };

  const observer = new IntersectionObserver(
    () => {
      // Measure every step against the band rather than ranking the entries in
      // this batch. Two things go wrong otherwise, and a long step hits both:
      // a batch only carries the steps whose visibility just changed, so the
      // step now filling the band is often absent from it, and
      // intersectionRatio is a fraction of the step's own height, so a step
      // more than ~5 band-heights tall never crosses a threshold above 0. Such
      // a step gets one callback as it enters, loses the comparison to the
      // step it is replacing (which is still in the band, and shorter, so
      // scores higher), and never gets another callback to win on — the reader
      // scrolls past it dimmed until the step after it takes over. Coverage is
      // measured against the band, so height does not distort it.
      let best = -1;
      let bestCoverage = 0;
      steps.forEach((el, i) => {
        const coverage = bandCoverage(el);
        if (coverage > bestCoverage) {
          best = i;
          bestCoverage = coverage;
        }
      });
      // Nothing in the band (between sections, or past the end): keep the
      // step the reader last had.
      if (best >= 0) setActive(best);
    },
    {rootMargin: `${-BAND_INSET_PCT}% 0px`, threshold: [0, 0.25, 0.5, 0.75, 1]}
  );

  steps.forEach((el) => observer.observe(el));

  // Steps are also directly selectable. Scrolling is the primary interaction,
  // but clicking or tabbing to a step and pressing Enter/Space jumps to it —
  // which keyboard users need, and which gives the page a usable fallback if
  // the observer never fires.
  const onClick = (e) => {
    const el = e.target.closest(".step");
    if (el) setActive(steps.indexOf(el));
  };
  const onKey = (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const el = e.target.closest(".step");
    if (!el) return;
    e.preventDefault();
    setActive(steps.indexOf(el));
  };

  steps.forEach((el) => {
    el.tabIndex = 0;
    el.setAttribute("role", "button");
  });
  root.addEventListener("click", onClick);
  root.addEventListener("keydown", onKey);

  setActive(0);

  return () => {
    observer.disconnect();
    root.removeEventListener("click", onClick);
    root.removeEventListener("keydown", onKey);
  };
}
