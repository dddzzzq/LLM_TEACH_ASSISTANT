(els, patterns) => {
          const clean=value=>String(value||'').replace(/\s+/g,' ').trim();
          const visible=e=>!!e.getClientRects().length && getComputedStyle(e).visibility!=='hidden' &&
            !e.closest('[hidden],[aria-hidden=true]');
          const labelled=e=>clean(e.getAttribute('aria-label') ||
            (e.getAttribute('aria-labelledby')||'').split(/\s+/).map(id=>document.getElementById(id)?.textContent||'').join(' '));
          const sectionName=e=>labelled(e) || clean(e.querySelector(':scope > legend')?.textContent);
          // The same patterns drive grouping here and format/scope validation
          // in Go. Keep official translations such as "commit attachments"
          // in one place, rather than maintaining two diverging label lists.
          const matches=(kind,name)=>!!patterns?.[kind] && new RegExp(patterns[kind],'i').test(name);
          return els.flatMap((e,i)=>{
            if(!visible(e)) return [];
            if(!e.matches('label') && e.closest('label')?.querySelector('input[type=checkbox],input[type=radio]')) return [];
            const roleOwner=e.closest('[role=checkbox],[role=radio]');
            if(roleOwner && roleOwner!==e) return [];
            const owner=e.closest('label,li,.export-range,.out,.nameItem,[role=radio],[role=checkbox]')||e.parentElement;
            const name=labelled(e) || clean(owner?.innerText);
            let group='';
            for(let node=e; node && !node.matches('[role=dialog],.popDiv.centerPop'); node=node.parentElement) {
              if(node.matches('.nameItem')) {group='naming';break;}
              if(node.matches('.out')) {group='format';break;}
              if(node.matches('.export-range')) {group='scope';break;}
              if(node.matches('fieldset,[role=group],[role=radiogroup]')) {
                const heading=sectionName(node);
                if(/命名|文件名|naming|file\s*name/i.test(heading)) group='naming';
                else if(/导出格式|导出内容|附件格式|export\s+(?:format|content)|attachment\s+format/i.test(heading)) group='format';
                else if(/导出范围|班级范围|导出班级|选择班级|export\s+(?:range|scope)|class\s+(?:range|scope)|select\s+class/i.test(heading)) group='scope';
                if(group) break;
              }
            }
            // Compatibility with simple labels without fieldsets. Recognize
            // actual option labels, never an incidental "班级" in a description.
            if(!group && /默认格式|文件原名|命名|学号.*姓名|default\s+format|original\s+file\s*name|naming/i.test(name)) group='naming';
            if(!group && (matches('attachments',name)||matches('other_format',name))) group='format';
            if(!group && (matches('current_class',name)||matches('all_classes',name))) group='scope';
            if(!group && /^(班级|教学班|行政班|class\s+)[A-Za-z0-9一二三四五六七八九十（( -]/i.test(name)) group='scope';
            return [{i,name,group,all:group==='scope' &&
              (!!e.closest('.all') || matches('all_classes',name) || /^(全部|所有|all)$/i.test(name))}];
          });
        }
