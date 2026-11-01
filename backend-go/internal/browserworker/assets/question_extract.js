roots => {
  const output=[], excluded='script,style,noscript,button,input,select,textarea,nav,header,footer,'+
    '[hidden],[aria-hidden=true],.mark_answer,.mark_key,.mark_score,.answer,.analysis,'+
    '[data-answer],[data-student-answer],.student-answer,.mark_btn';
  const visible=e=>!!e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden'&&
    !e.closest(excluded);
  const clean=s=>String(s||'').replace(/\u00a0/g,' ').trim();
  let buffer='';
  const flush=()=>{const text=clean(buffer).replace(/\n[ \t]+/g,'\n').replace(/\n{3,}/g,'\n\n');
    if(text)output.push({kind:'text',text});buffer='';};
  const walk=node=>{
    if(node.nodeType===Node.TEXT_NODE){buffer+=node.textContent;return;}
    if(node.nodeType!==Node.ELEMENT_NODE||!visible(node))return;
    if(node.matches('img,svg,canvas')){
      flush(); const index=output.length;
      node.setAttribute('data-rpa-question-resource',String(index));
      output.push({kind:'image',name:clean(node.getAttribute('alt'))||'题目图片',
        url:node.currentSrc||node.getAttribute('src')||'',resource_index:index});return;
    }
    if(node.matches('a[href],object[data],iframe[src],embed[src]')){
      const url=node.href||node.data||node.src||'';
      const name=clean(node.innerText||node.getAttribute('title')||node.getAttribute('download'));
      if(node.matches('object,iframe,embed')||/\.(pdf|docx?|txt|md)(?:[?#]|$)/i.test(url)||/\.(pdf|docx?|txt|md)$/i.test(name)||
        /附件|下载/.test(name)||node.hasAttribute('download')){
        flush();const index=output.length;
        node.setAttribute('data-rpa-question-resource',String(index));
        output.push({kind:'attachment',name:name||'题目附件',url,resource_index:index});return;
      }
    }
    const block=/^(P|DIV|LI|TR|H[1-6]|SECTION|ARTICLE|PRE)$/.test(node.tagName);
    if(block&&buffer&&!buffer.endsWith('\n'))buffer+='\n';
    if(node.tagName==='BR')buffer+='\n';
    for(const child of node.childNodes)walk(child);
    if(node.tagName==='TD'||node.tagName==='TH')buffer+='\t';
    if(block)buffer+='\n';
  };
  const outer=roots.filter(e=>visible(e)&&!roots.some(p=>p!==e&&p.contains(e)));
  for(const root of outer){walk(root);flush();}
  return {sources:output.slice(0,41),truncated:output.length>40||output.some(s=>(s.text||'').length>100000)};
}