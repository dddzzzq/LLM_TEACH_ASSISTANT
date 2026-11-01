document.addEventListener('click', event => {
              const button=event.target.closest('a.confirmDown,button');
              const modal=button?.closest('.popDiv.centerPop,[role=dialog]');
              if(modal && /导出|\bexport\b/i.test(modal.innerText) && /^(?:确定|确认|confirm|ok|export|start export)$/i.test(button.innerText.trim()))
                window.__rpa_manual_export().catch(()=>{});
            }, true);
