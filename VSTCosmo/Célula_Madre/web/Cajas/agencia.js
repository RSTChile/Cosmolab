window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'agencia',tit:'🧭 Alteridad / Agencia',w:4,h:6,render:(b,r,bf)=>{b.innerHTML=
     cjRowT('alt_otro_presente',si(r.alt_otro_presente),'¿Hay otro organismo presente?')
    +cjRowG('alt_intencion_comunicativa',r.alt_intencion_comunicativa,'Intención comunicativa (presencia)')+cjGauge(r.alt_intencion_comunicativa,'#6db6ff')+cjSpark(bf.alt_intencion_comunicativa,'#6db6ff')
    +cjRowG('alt_efecto_sobre_otro',r.alt_efecto_sobre_otro,'Efecto de su conducta sobre el otro')
    +cjRowG('alt_contingencia_social',r.alt_contingencia_social,'Contingencia social')+cjGauge(Math.min(1,(+r.alt_contingencia_social||0)*8),'#64f0c8')+cjSpark(bf.alt_contingencia_social,'#64f0c8')
    +cjRowG('alt_agencia_otro',r.alt_agencia_otro,'Agencia atribuida al otro')+cjGauge(r.alt_agencia_otro,'#5fd38a')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">presencia sobrevive a shuffle · agencia debe COLAPSAR (≈0 hoy)</div>`;}}
);
