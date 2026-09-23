/*! The Game of Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Collapsible control-panel sections, shared by both apps.
 *
 * Each section's heading becomes a button that folds the section to its
 * title, so the controls people reach for first stay on screen and the
 * specialist ones are one click away rather than a long scroll down. A
 * section marked data-collapsed in the HTML starts folded.
 */
export function collapsibleSections(panel) {
  for (const section of panel.querySelectorAll("section.grp")) {
    const heading = section.querySelector(":scope > h2");
    if (!heading) throw new Error("a panel section has no heading to fold on");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "grp-toggle";
    button.textContent = heading.textContent.trim();
    heading.replaceChildren(button);

    const fold = (folded) => {
      section.classList.toggle("collapsed", folded);
      button.setAttribute("aria-expanded", String(!folded));
    };
    fold(section.hasAttribute("data-collapsed"));
    button.addEventListener("click", () =>
      fold(!section.classList.contains("collapsed")),
    );
  }
}
