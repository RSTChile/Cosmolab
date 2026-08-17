window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'alteridad',tit:'🗣 Alteridad / Intención',w:4,h:6,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('alt_intencion_comunicativa',r.alt_intencion_comunicativa,'Intención comunicativa')+cjGauge(r.alt_intencion_comunicativa,'#64f0c8')+cjSpark(bf.alt_intencion_comunicativa,'#64f0c8')
    +cjRowG('alt_efecto_sobre_otro',r.alt_efecto_sobre_otro,'Efecto de su conducta sobre el otro')+cjGauge(r.alt_efecto_sobre_otro,'#6db6ff')
    +cjRowG('alt_efecto_sobre_mi',r.alt_efecto_sobre_mi,'Efecto del otro sobre sí mismo')+cjGauge(Math.max(0,Number(r.alt_efecto_sobre_mi)),'#5fd38a')
    +cjRowT('alt_otro_presente',si(r.alt_otro_presente),'¿Hay otro organismo presente?')
    +cjRowT('alt_patron_emitido',r.alt_patron_emitido,'Patrón que emitió hacia el otro');}}
);
