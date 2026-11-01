e => {
                            const attrs=n=>[n.getAttribute('aria-label'),n.getAttribute('title'),n.className,n.getAttribute('onclick')].filter(Boolean).join(' ');
                            const parent=e.parentElement?.closest('a,button,[role=button],[onclick]');
                            return {description:attrs(e),text:(e.innerText||'').trim(),nested:!!parent && !!e.closest('#downloadcenter,.downloadCenter')?.contains(parent) && /关闭|close|hideDownload/i.test(attrs(parent))};
                        }