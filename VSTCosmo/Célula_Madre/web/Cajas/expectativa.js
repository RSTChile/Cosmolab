window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'expectativa',tit:'🔭 Expectativa',w:4,h:6,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('expectativa',r.expectativa,'Expectativa tras la voz')+cjGauge(Math.min(1,(+r.expectativa||0)*8),'#b58cff')+cjSpark(bf.expectativa,'#b58cff')
    +cjRowG('expectativa_confianza',r.expectativa_confianza,'Confianza en la expectativa')+cjGauge(r.expectativa_confianza,'#6db6ff')
    +cjRowG('expectativa_exploracion',r.expectativa_exploracion,'Exploración que dispara')+cjGauge(Math.min(1,(+r.expectativa_exploracion||0)*5),'#5fd38a')
    +cjRowG('expectativa_confirmaciones',r.expectativa_confirmaciones,'Veces que se cumplió')
    +cjRowG('expectativa_falsaciones',r.expectativa_falsaciones,'Veces que falló')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">¿vale la pena seguir explorando tras la voz? voz→expectativa→exploración</div>`;}}
);
