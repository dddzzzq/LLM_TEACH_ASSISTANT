// Accept only product-owned result routes from task receipts.
export function taskResultLink(value) {
  if (typeof value !== 'string') return '';
  if (/^(\/materials|\/(assignments)\/\d+|\/settings\/portal)$/.test(value)) return value;
  if (
    /^\/grade-homework\?workflow=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(
      value
    )
  )
    return value;
  if (/^\/assignments\/\d+\?[^#]+$/.test(value)) {
    const params = new URLSearchParams(value.slice(value.indexOf('?') + 1));
    const seen = new Set();
    for (const [key, entry] of params) {
      if (seen.has(key)) return '';
      seen.add(key);
      if (key === 'tab') {
        if (!['content', 'materials', 'results'].includes(entry)) return '';
      } else if (key === 'class_id') {
        // Portal IDs and the product's local:<base64url> read scopes only.
        if (!/^(?:[A-Za-z0-9_-]{1,128}|local:[A-Za-z0-9_-]{1,1024})$/.test(entry)) return '';
      } else return '';
    }
    if (seen.size) return value;
  }
  return '';
}
