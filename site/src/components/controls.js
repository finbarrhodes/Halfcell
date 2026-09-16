// Choice controls in the site's own style, lighter than Inputs.radio.
//
// choiceGroup returns an element with a `value` property that dispatches
// "input" when the reader picks an option, so it works with view() and
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
    root.append(button);
    return button;
  });

  function select(option, notify) {
    current = option;
    buttons.forEach((button, i) => {
      const on = options[i] === option;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-checked", String(on));
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
