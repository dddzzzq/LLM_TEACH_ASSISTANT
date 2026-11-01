e => {
                  const menu=e.closest('ul.morePop');
                  for(let node=menu?.parentElement; node; node=node.parentElement) {
                    const style=getComputedStyle(node);
                    if(node.hidden || node.getAttribute('aria-hidden')==='true' ||
                       style.display==='none' || style.visibility==='hidden') return false;
                  }
                  return !!menu;
                }