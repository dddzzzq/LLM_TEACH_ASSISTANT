e => {
            if(!e.isConnected || !e.getClientRects().length || e.disabled ||
               e.closest('[hidden],[aria-hidden=true],[aria-disabled=true],.disabled,.is-disabled')) return false;
            e.click(); return true;
        }