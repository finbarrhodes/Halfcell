// Scroll-driven step tracking.
//
// Observes .step elements and reports the index of the one currently holding
// the reader's attention. Uses a viewport band rather than a single threshold
// so the active step changes at a predictable point regardless of step height,
// and falls back to "all steps visible" behaviour when the observer is
// unavailable or the reader prefers reduced motion.

export function watchSteps(root, onChange) {
  const steps = Array.from(root.querySelectorAll(".step"));
  if (!steps.length) return () => {};

  let active = -1;
  const setActive = (i) => {
    if (i === active) return;
    active = i;
    steps.forEach((el, j) => el.classList.toggle("is-active", j === i));
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
