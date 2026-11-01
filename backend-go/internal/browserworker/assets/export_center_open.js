e => {
                    if(!e.isConnected || !e.getClientRects().length ||
                       getComputedStyle(e).visibility==='hidden' || e.disabled ||
                       e.closest('[hidden],[aria-hidden=true],[aria-disabled=true],[inert],.disabled,.is-disabled')) return false;
                    e.click(); return true;
                }