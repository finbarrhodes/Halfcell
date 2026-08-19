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

  const observer = new IntersectionObserver(
    (entries) => {
      // Pick the visible step nearest the band's centre, so fast scrolling
      // lands somewhere sensible rather than on whichever fired last.
      const visible = entries
        .filter((e) => e.isIntersecting)
        .map((e) => ({i: steps.indexOf(e.target), ratio: e.intersectionRatio}));
      if (!visible.length) return;
      visible.sort((a, b) => b.ratio - a.ratio || a.i - b.i);
      setActive(visible[0].i);
    },
    // Band across the middle of the viewport
    {rootMargin: "-40% 0px -40% 0px", threshold: [0, 0.25, 0.5, 0.75, 1]}
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
