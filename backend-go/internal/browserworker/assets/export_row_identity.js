e => ({text:(e.innerText||'').trim(),
            platform_id:e.getAttribute('data-id')||e.getAttribute('data-taskid')||e.id||''})