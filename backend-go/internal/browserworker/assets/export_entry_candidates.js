(elements, {pattern, hidden}) => {
  const labels = new RegExp(pattern, 'i');
  const label = e => (e.getAttribute('aria-label') || e.innerText || e.textContent || e.value || '').trim().replace(/\s+/g, ' ');
  const visible = e => !!e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden' && !e.closest('[hidden],[aria-hidden=true],[inert]');
  const eligible = e => {
    if (!e.isConnected || e.disabled || e.closest('[aria-disabled=true],.disabled,.is-disabled') || !labels.test(label(e))) return false;
    if (!hidden) return visible(e);
    // Retain only the original portal's own hidden More menu compatibility.
    const menu = e.closest('ul.morePop');
    if (!menu || visible(e)) return false;
    for (let node = menu.parentElement; node; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (node.hidden || node.getAttribute('aria-hidden') === 'true' || node.hasAttribute('inert') || style.display === 'none' || style.visibility === 'hidden') return false;
    }
    return true;
  };
  const candidates = elements.map((element, index) => ({element, index})).filter(({element}) => eligible(element));
  // A menuitem containing an anchor is one action; two sibling anchors remain
  // ambiguous and must never be resolved by taking the first DOM match.
  return candidates.filter(({element}) => !candidates.some(other => other.element !== element && element.contains(other.element))).map(({index}) => index);
}
