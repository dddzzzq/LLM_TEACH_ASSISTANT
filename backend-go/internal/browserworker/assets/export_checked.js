e => {
          if(e.matches('input[type=checkbox],input[type=radio]')) return e.checked;
          const parent=e.closest('label,li,.export-range,.out,[role=radio],[role=checkbox]');
          const inputs=(parent||e).querySelectorAll('input[type=checkbox],input[type=radio]');
          if(inputs.length===1) return inputs[0].checked;
          for(const node of [e,parent].filter(Boolean)) {
            const state=node.getAttribute('aria-checked') ?? node.getAttribute('data-checked');
            if(state!==null) return state==='true' || state==='1';
          }
          const selected=/(^|\s)(checked|active|selected|check_on|on|grade_checked|check_answer)(\s|$)/;
          return [e,parent].filter(Boolean).some(node=>selected.test(String(node.className)));
        }