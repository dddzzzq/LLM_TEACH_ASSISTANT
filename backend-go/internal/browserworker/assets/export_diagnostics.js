els => els.slice(0,80).map(e=>({
                        tag:e.tagName,id:e.id,classes:String(e.className).slice(0,80),
                        text:(e.textContent||'').trim().slice(0,80),
                        attributes:[...e.attributes]
                            .filter(a=>/^(?:value|name|(?:data-)?(?:class|clazz)[\w-]*|data-(?:value|id))$/i.test(a.name)&&a.name!=='class')
                            .map(a=>[a.name,a.value.slice(0,80)]),
                        handler:(e.getAttribute('onclick')||'').match(/^\s*([\w.]+)/)?.[1]||'',
                        numeric_arguments:[...(e.getAttribute('onclick')||'')
                            .matchAll(/(?:^|[(,])\s*["']?(\d+)["']?(?=\s*[,)]|$)/g)]
                            .map(m=>m[1]).slice(0,8),
                        numeric_attributes:[...e.attributes]
                            .filter(a=>/^\d+$/.test(a.value)&&!/token|secret|password|enc|cookie/i.test(a.name))
                            .map(a=>[a.name,a.value]),
                        numeric_children:[...e.querySelectorAll('input')]
                            .filter(n=>/^\d+$/.test(n.value)&&!/token|secret|password|enc|cookie/i.test(n.name))
                            .map(n=>({name:n.name,value:n.value})).slice(0,8)
                    }))