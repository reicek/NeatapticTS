(function initializeThemeTooltips() {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return;
  }

  if (window.__NEATAPTIC_THEME_TOOLTIPS_READY__) {
    return;
  }

  window.__NEATAPTIC_THEME_TOOLTIPS_READY__ = true;

  const TOOLTIP_BODY_ATTRIBUTE = 'data-theme-tooltip-body';
  const TOOLTIP_TITLE_ATTRIBUTE = 'data-tooltip-title';
  const PRESERVED_TITLE_ATTRIBUTE = 'data-theme-tooltip-native-title';
  const ROLE_TOOLTIP_ID_ATTRIBUTE = 'data-theme-tooltip-id';
  const TOOLTIP_CONTENT_SELECTOR = `[${TOOLTIP_BODY_ATTRIBUTE}]`;
  const TOOLTIP_FADE_DURATION_MS = 180;

  let activeAnchorElement = null;
  let hideTooltipTimeoutId = null;
  let tooltipElement = null;
  let tooltipTitleElement = null;
  let tooltipBodyElement = null;

  ensureTooltipElement();
  hydrateTooltipSources();
  observeDynamicTooltipSources();

  document.addEventListener('pointerover', onPointerOver, true);
  document.addEventListener('pointerout', onPointerOut, true);
  document.addEventListener('pointermove', onPointerMove, true);
  document.addEventListener('focusin', onFocusIn, true);
  document.addEventListener('focusout', onFocusOut, true);
  window.addEventListener(
    'scroll',
    () => {
      if (activeAnchorElement) {
        positionTooltip(activeAnchorElement);
      }
    },
    true,
  );
  window.addEventListener('resize', () => {
    if (activeAnchorElement) {
      positionTooltip(activeAnchorElement);
    }
  });

  function onPointerOver(event) {
    const eventTarget = event.target;
    if (!(eventTarget instanceof Element)) {
      return;
    }

    const tooltipAnchor = eventTarget.closest(TOOLTIP_CONTENT_SELECTOR);
    const titledAnchor = eventTarget.closest('[title]');
    if (!tooltipAnchor && titledAnchor instanceof HTMLElement) {
      hydrateTitleElement(titledAnchor);
    }

    const resolvedTooltipAnchor = eventTarget.closest(TOOLTIP_CONTENT_SELECTOR);
    if (!resolvedTooltipAnchor) {
      return;
    }

    showTooltip(resolvedTooltipAnchor);
  }

  function onPointerOut(event) {
    const eventTarget = event.target;
    if (!(eventTarget instanceof Element)) {
      return;
    }

    if (!activeAnchorElement) {
      return;
    }

    const relatedTarget = event.relatedTarget;
    if (
      relatedTarget instanceof Node &&
      activeAnchorElement.contains(relatedTarget)
    ) {
      return;
    }

    if (eventTarget.closest(TOOLTIP_CONTENT_SELECTOR) === activeAnchorElement) {
      hideTooltip();
    }
  }

  function onPointerMove(event) {
    if (!activeAnchorElement) {
      return;
    }

    if (!(event.target instanceof Node)) {
      hideTooltip();
      return;
    }

    if (!document.contains(activeAnchorElement)) {
      hideTooltip(true);
      return;
    }

    if (!activeAnchorElement.contains(event.target)) {
      hideTooltip();
    }
  }

  function onFocusIn(event) {
    const eventTarget = event.target;
    if (!(eventTarget instanceof Element)) {
      return;
    }

    const tooltipAnchor = eventTarget.closest(TOOLTIP_CONTENT_SELECTOR);
    if (!tooltipAnchor && eventTarget instanceof HTMLElement) {
      hydrateTitleElement(eventTarget);
    }

    const resolvedTooltipAnchor = eventTarget.closest(TOOLTIP_CONTENT_SELECTOR);
    if (!resolvedTooltipAnchor) {
      return;
    }

    showTooltip(resolvedTooltipAnchor);
  }

  function onFocusOut(event) {
    const eventTarget = event.target;
    if (!(eventTarget instanceof Element)) {
      return;
    }

    if (!activeAnchorElement) {
      return;
    }

    const relatedTarget = event.relatedTarget;
    if (
      relatedTarget instanceof Node &&
      activeAnchorElement.contains(relatedTarget)
    ) {
      return;
    }

    if (eventTarget.closest(TOOLTIP_CONTENT_SELECTOR) === activeAnchorElement) {
      hideTooltip();
    }
  }

  function showTooltip(anchorElement) {
    ensureTooltipElement();

    if (hideTooltipTimeoutId !== null) {
      window.clearTimeout(hideTooltipTimeoutId);
      hideTooltipTimeoutId = null;
    }

    const tooltipBody = anchorElement.getAttribute(TOOLTIP_BODY_ATTRIBUTE);
    if (!tooltipBody) {
      hideTooltip(true);
      return;
    }

    const explicitTooltipTitle = anchorElement.getAttribute(
      TOOLTIP_TITLE_ATTRIBUTE,
    );
    const fallbackTooltipTitle = deriveTooltipTitle(anchorElement);
    const tooltipTitle = (
      explicitTooltipTitle ||
      fallbackTooltipTitle ||
      'Details'
    ).trim();

    activeAnchorElement = anchorElement;
    tooltipTitleElement.textContent = tooltipTitle;
    tooltipBodyElement.textContent = tooltipBody;
    tooltipElement.hidden = false;

    ensureTooltipAnchorBinding(anchorElement);
    positionTooltip(anchorElement);

    window.requestAnimationFrame(() => {
      if (activeAnchorElement === anchorElement) {
        tooltipElement.dataset.visible = 'true';
      }
    });
  }

  function hideTooltip(forceHide) {
    if (!tooltipElement) {
      return;
    }

    tooltipElement.dataset.visible = 'false';
    activeAnchorElement = null;

    if (hideTooltipTimeoutId !== null) {
      window.clearTimeout(hideTooltipTimeoutId);
      hideTooltipTimeoutId = null;
    }

    if (forceHide === true) {
      tooltipElement.hidden = true;
      return;
    }

    hideTooltipTimeoutId = window.setTimeout(() => {
      if (tooltipElement.dataset.visible !== 'true') {
        tooltipElement.hidden = true;
      }
      hideTooltipTimeoutId = null;
    }, TOOLTIP_FADE_DURATION_MS);
  }

  function positionTooltip(anchorElement) {
    if (!tooltipElement || tooltipElement.hidden) {
      return;
    }

    const anchorRect = anchorElement.getBoundingClientRect();
    const tooltipRect = tooltipElement.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const horizontalMargin = 12;
    const verticalOffset = 12;

    let top = anchorRect.top - tooltipRect.height - verticalOffset;
    let left = anchorRect.left + (anchorRect.width - tooltipRect.width) / 2;

    if (left < horizontalMargin) {
      left = horizontalMargin;
    }

    if (left + tooltipRect.width > viewportWidth - horizontalMargin) {
      left = viewportWidth - tooltipRect.width - horizontalMargin;
    }

    if (top < horizontalMargin) {
      top = anchorRect.bottom + verticalOffset;
    }

    if (top + tooltipRect.height > viewportHeight - horizontalMargin) {
      top = Math.max(
        horizontalMargin,
        viewportHeight - tooltipRect.height - horizontalMargin,
      );
    }

    tooltipElement.style.left = `${left}px`;
    tooltipElement.style.top = `${top}px`;
  }

  function hydrateTooltipSources() {
    const titledElements = document.querySelectorAll('[title]');

    titledElements.forEach((element) => {
      if (element instanceof HTMLElement) {
        hydrateTitleElement(element);
      }
    });
  }

  function observeDynamicTooltipSources() {
    if (typeof MutationObserver === 'undefined') {
      return;
    }

    const observer = new MutationObserver((mutationRecords) => {
      for (const mutationRecord of mutationRecords) {
        if (mutationRecord.type === 'attributes') {
          const targetElement = mutationRecord.target;
          if (targetElement instanceof HTMLElement) {
            hydrateTitleElement(targetElement);
          }
          continue;
        }

        for (const addedNode of mutationRecord.addedNodes) {
          if (!(addedNode instanceof HTMLElement)) {
            continue;
          }

          if (addedNode.hasAttribute('title')) {
            hydrateTitleElement(addedNode);
          }

          addedNode.querySelectorAll('[title]').forEach((nestedElement) => {
            if (nestedElement instanceof HTMLElement) {
              hydrateTitleElement(nestedElement);
            }
          });
        }
      }
    });

    observer.observe(document.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['title'],
    });
  }

  function hydrateTitleElement(element) {
    const titleText = element.getAttribute('title');
    if (!titleText || !titleText.trim()) {
      return;
    }

    element.setAttribute(PRESERVED_TITLE_ATTRIBUTE, titleText);
    element.setAttribute(TOOLTIP_BODY_ATTRIBUTE, titleText.trim());
    element.removeAttribute('title');
  }

  function deriveTooltipTitle(anchorElement) {
    const ariaLabel = anchorElement.getAttribute('aria-label');
    if (ariaLabel && ariaLabel.trim()) {
      return ariaLabel.trim();
    }

    const ownText = (anchorElement.textContent || '')
      .replace(/\s+/g, ' ')
      .trim();
    if (ownText) {
      return ownText;
    }

    const nearestTableHeader = anchorElement.closest('th');
    if (nearestTableHeader) {
      const headerText = (nearestTableHeader.textContent || '')
        .replace(/\s+/g, ' ')
        .trim();
      if (headerText) {
        return `${headerText} column`;
      }
    }

    const tagName = anchorElement.tagName.toLowerCase();
    if (tagName === 'td') {
      return 'Table value';
    }
    if (tagName === 'button') {
      return 'Button';
    }

    return 'Details';
  }

  function ensureTooltipElement() {
    if (tooltipElement) {
      return;
    }

    tooltipElement = document.createElement('div');
    tooltipElement.className = 'theme-tooltip';
    tooltipElement.setAttribute('role', 'tooltip');
    tooltipElement.dataset.visible = 'false';
    tooltipElement.hidden = true;

    tooltipTitleElement = document.createElement('div');
    tooltipTitleElement.className = 'theme-tooltip-title';

    tooltipBodyElement = document.createElement('div');
    tooltipBodyElement.className = 'theme-tooltip-body';

    tooltipElement.appendChild(tooltipTitleElement);
    tooltipElement.appendChild(tooltipBodyElement);
    document.body.appendChild(tooltipElement);
  }

  function ensureTooltipAnchorBinding(anchorElement) {
    const existingId = anchorElement.getAttribute(ROLE_TOOLTIP_ID_ATTRIBUTE);
    if (existingId) {
      return;
    }

    const generatedId = `theme-tooltip-anchor-${Math.random().toString(36).slice(2)}`;
    anchorElement.setAttribute(ROLE_TOOLTIP_ID_ATTRIBUTE, generatedId);
  }
})();
