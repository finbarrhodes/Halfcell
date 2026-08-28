// Sidebar section menu.
//
// Inlined into every page's <head> by observablehq.config.js rather than
// imported from a page, so it applies site-wide without touching each .md.
//
// Two jobs:
//
// 1. Unwrap Framework's heading self-links. Framework wraps every h1-h4 that
//    carries an id in <a class="observablehq-header-anchor" href="#same-id">,
//    hardcoded in dist/html.js after markdown rendering — no config option
//    reaches it. The link points at the heading you are already looking at: it
//    costs a tab stop per heading, underlines every heading on hover, and
//    navigates nowhere. Unwrapping leaves the id in place, which is what the
//    menu below actually needs.
//
// 2. Build a menu of the current page's sections under its sidebar entry, and
//    keep the section you are reading marked. That is the navigation the
//    self-links gestured at without ever providing.

const main = document.querySelector("#observablehq-main");
const active = document.querySelector("#observablehq-sidebar .observablehq-link-active");

for (const a of document.querySelectorAll("a.observablehq-header-anchor")) {
  a.replaceWith(...a.childNodes);
}

// Methodology is flat — every heading on it is an h2 — so a menu there is a
// twelve-item list with no hierarchy to lean on, taller than the page nav above
// it and no faster to scan than the page. It opts out; the anchor unwrap above
// still applies to it.
//
// Normalised because the path is extensionless on Cloudflare Pages and under
// `observable preview`, but keeps .html when a built file is opened directly.
const NO_SECTION_MENU = new Set(["/methodology"]);
const path = location.pathname.replace(/\.html$/, "").replace(/\/$/, "") || "/";

// Top-level sections only. Including h3 gave 12 and 17 entries on the two chart
// pages — long enough that the menu stopped being scannable, which is the only
// thing it is for. The h3s are still reachable by scrolling the section.
//
// Only headings markdown gave an id, which already excludes the raw-HTML
// headings inside cards and KPI tiles. Nothing generates an h2 inside a step or
// figure today; the filter keeps it that way if that changes.
const sections =
  main && !NO_SECTION_MENU.has(path)
    ? [...main.querySelectorAll("h2[id]")].filter((h) => !h.closest(".step, .card, figure"))
    : [];

if (active && sections.length) {
  const menu = document.createElement("ol");
  menu.className = "hc-subnav";

  const items = new Map();
  for (const heading of sections) {
    const item = document.createElement("li");
    item.className = `hc-subnav-item hc-subnav-${heading.tagName.toLowerCase()}`;
    const link = document.createElement("a");
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent.trim();
    item.append(link);
    menu.append(item);
    items.set(heading, item);
  }
  active.append(menu);

  // Mark the last section whose heading has passed the top of the viewport.
  // Nothing is marked above the first heading — on Forecasting & Dispatch the
  // walkthrough runs before the first h2, and claiming a section there would be
  // wrong.
  const LINE = 140;
  let queued = false;

  const sync = () => {
    queued = false;
    let current = null;
    for (const heading of sections) {
      if (heading.getBoundingClientRect().top > LINE) break;
      current = heading;
    }
    // The final section can never reach the line on a short last block.
    if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4) {
      current = sections[sections.length - 1];
    }
    for (const [heading, item] of items) item.classList.toggle("is-current", heading === current);
  };

  const schedule = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(sync);
  };

  addEventListener("scroll", schedule, {passive: true});
  addEventListener("resize", schedule, {passive: true});
  // A frame requested while the tab was hidden may never run, which leaves
  // `queued` stuck true and silently kills the spy. Clear it and sync directly
  // on the way back rather than going through schedule().
  addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    queued = false;
    sync();
  });
  sync();

  // Below Framework's 1008px breakpoint the sidebar is an overlay, so leaving it
  // open would cover the section just jumped to.
  menu.addEventListener("click", (event) => {
    if (!event.target.closest("a")) return;
    const toggle = document.querySelector("#observablehq-sidebar-toggle");
    if (toggle && !matchMedia("(min-width: 1008px)").matches) toggle.checked = false;
  });
}
