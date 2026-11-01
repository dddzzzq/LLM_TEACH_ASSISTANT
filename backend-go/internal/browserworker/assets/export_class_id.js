e => {
              const row=e.closest('[data-clazzid],[data-classid],[clazzid],[classid]');
              const input=e.matches('input')?e:e.querySelector('input');
              return e.getAttribute('data-clazzid')||e.getAttribute('data-classid')||
                row?.getAttribute('data-clazzid')||row?.getAttribute('data-classid')||
                row?.getAttribute('clazzid')||row?.getAttribute('classid')||
                (/^(clazzid|classid)$/i.test(input?.name||'')?input.value:'')||'';
            }