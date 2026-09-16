// Form controls in the site's own style, lighter than Framework's Inputs.
//
// Every control here returns an element with a `value` property that dispatches
// "input" when the reader changes it, so it works with view() and
// Generators.input() exactly like the built-in Inputs.

export function choiceGroup(options, {value, format = String, label} = {}) {
  const root = document.createElement("div");
  root.className = "choice-group";
  root.setAttribute("role", "radiogroup");
  if (label) root.setAttribute("aria-label", label);

  let current = options.includes(value) ? value : options[0];

  const buttons = options.map((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "choice";
    button.setAttribute("role", "radio");

    button.textContent = format(option);

    button.addEventListener("click", () => select(option, true));
    button.addEventListener("keydown", (event) => {
      // Arrow keys move within a radiogroup; Tab leaves it. Native radios do
      // this for free, and these buttons stand in for native radios.
      const step = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[event.key];
      if (!step) return;
      event.preventDefault();
      const next = (options.indexOf(current) + step + options.length) % options.length;
      select(options[next], true);
      buttons[next].focus();
    });
    root.append(button);
    return button;
  });

  function select(option, notify) {
    current = option;
    buttons.forEach((button, i) => {
      const on = options[i] === option;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-checked", String(on));
      // One tab stop for the whole group, landing on the current choice.
      button.tabIndex = on ? 0 : -1;
    });
    if (notify) root.dispatchEvent(new Event("input", {bubbles: true}));
  }

  Object.defineProperty(root, "value", {
    get: () => current,
    set: (option) => select(option, false),
  });

  select(current, false);
  return root;
}

// A range slider with a typed, unit-suffixed readout beside it.
//
// Inputs.range puts a bare number box first and leaves both halves at browser
// defaults. Here the slider leads — it is the control people actually drag —
// and the box doubles as the readout for exact entry.
export function slider({min = 0, max = 100, step = 1, value = min, unit = "", label} = {}) {
  const root = document.createElement("div");
  root.className = "control-slider";

  const range = document.createElement("input");
  range.type = "range";
  range.min = min;
  range.max = max;
  range.step = step;
  if (label) range.setAttribute("aria-label", label);

  const number = document.createElement("input");
  number.type = "number";
  number.className = "control-number";
  number.min = min;
  number.max = max;
  number.step = step;
  number.inputMode = "numeric";
  if (label) number.setAttribute("aria-label", label);

  const box = document.createElement("div");
  box.className = "control-number-box";
  box.append(number);
  if (unit) {
    const suffix = document.createElement("span");
    suffix.className = "control-unit";
    suffix.textContent = unit;
    box.append(suffix);
  }

  let current = Math.min(max, Math.max(min, Number(value)));

  function set(next, notify) {
    if (!Number.isFinite(next)) return;
    const clamped = Math.min(max, Math.max(min, next));
    range.value = String(clamped);
    if (Number(number.value) !== clamped) number.value = String(clamped);
    if (clamped === current) return;
    current = clamped;
    if (notify) root.dispatchEvent(new Event("input", {bubbles: true}));
  }

  // Inner events are swallowed and re-dispatched from the root, so a half-typed
  // number ("5" on the way to "50") never reaches the page as a real value.
  range.addEventListener("input", (event) => {
    event.stopPropagation();
    set(range.valueAsNumber, true);
  });
  number.addEventListener("input", (event) => {
    event.stopPropagation();
    const typed = number.valueAsNumber;
    if (Number.isFinite(typed) && typed >= min && typed <= max) set(typed, true);
  });
  // Out-of-range or empty entries are only corrected on commit, so the box does
  // not rewrite itself under the cursor.
  number.addEventListener("change", (event) => {
    event.stopPropagation();
    set(Number.isFinite(number.valueAsNumber) ? number.valueAsNumber : current, true);
    number.value = String(current);
  });

  Object.defineProperty(root, "value", {
    get: () => current,
    set: (next) => set(Number(next), false),
  });

  root.append(range, box);
  set(current, false);
  return root;
}

const isoDay = (d) => new Date(d).toISOString().slice(0, 10);

// A pair of date fields read as one range. Returns the wrapper to display, with
// `from` and `to` as the two value-bearing elements to observe separately —
// each yields a Date, as Inputs.date does, so callers keep UTC month maths.
export function dateRange({value: [from, to] = [], min, max} = {}) {
  const root = document.createElement("div");
  root.className = "control-dates";

  const separator = document.createElement("span");
  separator.className = "control-date-sep";
  separator.textContent = "–";

  const fromField = dateField({value: from, min, max, label: "From"});
  const toField = dateField({value: to, min, max, label: "To"});

  root.append(fromField, separator, toField);
  return Object.assign(root, {from: fromField, to: toField});
}

function dateField({value, min, max, label}) {
  const root = document.createElement("div");
  root.className = "control-date";

  const input = document.createElement("input");
  input.type = "date";
  if (min != null) input.min = isoDay(min);
  if (max != null) input.max = isoDay(max);
  if (label) input.setAttribute("aria-label", label);
  input.value = isoDay(value);

  let current = input.valueAsDate;

  // A date input reads empty mid-edit and while the picker is open. Holding the
  // last valid date keeps the charts from blanking out between keystrokes.
  input.addEventListener("input", () => {
    if (input.valueAsDate) current = input.valueAsDate;
  });

  Object.defineProperty(root, "value", {
    get: () => current,
    set: (next) => {
      current = new Date(next);
      input.value = isoDay(current);
    },
  });

  root.append(input);
  return root;
}

// Lays a set of labelled controls out as one panel, so a page's inputs read as
// a single instrument rather than a stack of loose form rows.
export function controlPanel(fields, {title} = {}) {
  const root = document.createElement("div");
  root.className = "controls";

  if (title) {
    const heading = document.createElement("div");
    heading.className = "controls-title";
    heading.textContent = title;
    root.append(heading);
  }

  const grid = document.createElement("div");
  grid.className = "controls-grid";
  for (const {label, input} of fields) {
    const field = document.createElement("div");
    field.className = "control-field";
    const caption = document.createElement("span");
    caption.className = "control-label";
    caption.textContent = label;
    field.append(caption, input);
    grid.append(field);
  }

  root.append(grid);
  return root;
}
