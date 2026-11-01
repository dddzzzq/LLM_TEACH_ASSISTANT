(element, {pattern, legacy}) => {
  const name = (element.getAttribute('aria-label') || element.innerText || element.textContent || element.value || '').trim().replace(/\s+/g, ' ');
  if (!element.isConnected || typeof element.click !== 'function' || !new RegExp(pattern, 'i').test(name) ||
      element.disabled || element.closest('[aria-disabled=true],[inert],.disabled,.is-disabled')) return false;
  const visible = !!element.getClientRects().length && getComputedStyle(element).visibility !== 'hidden' &&
    !element.closest('[hidden],[aria-hidden=true]');
  if (!visible) {
    // Only the previously verified, site-owned legacy More menu may be hidden.
    // Hidden course views and arbitrary hidden controls are never activated.
    const menu = legacy && element.closest('ul.morePop');
    if (!menu) return false;
    for (let node = menu.parentElement; node; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (node.hidden || node.getAttribute('aria-hidden') === 'true' || node.hasAttribute('inert') ||
          style.display === 'none' || style.visibility === 'hidden') return false;
    }
  }
  // Preserve the original Python script's native menu activation. Moving the
  // physical pointer can close a hover menu before Playwright finishes clicking.
  element.click();
  return true;
}
