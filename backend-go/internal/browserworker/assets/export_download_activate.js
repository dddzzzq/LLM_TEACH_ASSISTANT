(button,{row,text,platform_id}) => {
            if(!row.isConnected || !button.isConnected || !row.contains(button) ||
               (row.innerText||'').trim()!==text ||
               (row.getAttribute('data-id')||row.getAttribute('data-taskid')||row.id||'')!==platform_id ||
               !button.getClientRects().length || getComputedStyle(button).visibility==='hidden' ||
               button.closest('[hidden],[aria-hidden=true],[aria-disabled=true],[inert],.disabled,.is-disabled') ||
               button.disabled) return false;
            button.click();
            return true;
        }