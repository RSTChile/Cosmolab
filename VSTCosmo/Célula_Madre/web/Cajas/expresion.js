window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'expresion',tit:'🎙 Expresión vocal (conducta)',w:4,h:7,render:(b,r,bf)=>{const voc=(+r.expr_vocalizando||0)>=.5,sil=(+r.expr_silencio||0)>=.5;
     const vb=(bf.expr_vocalizando||[]).slice(-150).map(Number).filter(x=>x===x);
     const fh=vb.length?vb.filter(x=>x>=.5).length/vb.length:0;
     const medidor=`<div class="obsrow"><span class="obsk">🗣 Habla</span><span class="obsv">${(fh*100).toFixed(0)}% · No Habla ${((1-fh)*100).toFixed(0)}% 🤫</span></div>`
       +`<div class="obsgauge" style="display:flex"><i style="width:${(fh*100).toFixed(1)}%;background:#5fd38a"></i><i style="width:${((1-fh)*100).toFixed(1)}%;background:#8aa0b8"></i></div>`;
     b.innerHTML=
     cjRowT('expr_vocalizando', voc?'🗣 VOCALIZA':(sil?'🤫 SILENCIO':'·'),'Conducta vocal actual')+medidor
    +cjRowG('expr_p_voz',r.expr_p_voz,'Probabilidad de vocalizar')+cjGauge(r.expr_p_voz,'#e8b86d')+cjSpark(bf.expr_p_voz,'#e8b86d')
    +cjRowG('expr_long_conducta',r.expr_long_conducta,'Duración de la conducta vocal')
    +cjRowG('expr_long_silencio',r.expr_long_silencio,'Duración del silencio')
    +cjRowG('expr_recurso',r.expr_recurso,'Recurso disponible para hablar')+cjGauge(Math.max(0,Math.min(1,+r.expr_recurso||0)),'#5fd38a')
    +cjRowG('expr_peso_silencio',r.expr_peso_silencio,'Peso del silencio')
    +cjRowG('expr_familiaridad',r.expr_familiaridad,'Familiaridad de la región')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">el 1er acto es decidir SI hablar; silencio y voz compiten por historia</div>`;}}
);
