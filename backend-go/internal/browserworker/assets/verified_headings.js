els => els.filter(e => e.getClientRects().length &&
  getComputedStyle(e).visibility !== 'hidden' &&
  !e.closest('[hidden],[aria-hidden=true],li,tr,[role=listitem],[role=row],[role=list]'))
  .map(e => (e.innerText || '').trim())
