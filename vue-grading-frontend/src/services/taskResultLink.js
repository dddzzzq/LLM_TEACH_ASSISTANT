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
  return '';
}
