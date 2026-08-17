window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'aprendizaje',tit:'🧠 Aprendizaje (OAO: ecoica + imitación)',w:4,h:6,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('oao_oido',r.oao_oido,'¿Oye al otro?')+cjGauge(Math.min(1,(+r.oao_oido||0)*3),'#6db6ff')
    +cjRowG('oao_echoica_n',r.oao_echoica_n,'Trazas en la memoria ecoica')+cjGauge(Math.min(1,(+r.oao_echoica_n||0)/100),'#b58cff')
    +cjRowG('oao_imitacion_mag',r.oao_imitacion_mag,'Magnitud de la imitación')+cjGauge(Math.min(1,(+r.oao_imitacion_mag||0)*2),'#8ef0c0')+cjSpark(bf.oao_imitacion_mag,'#8ef0c0')
    +cjRowT('oao_aprendio',si(r.oao_aprendio),'¿Incorporó lo oído?')
    +cjRowG('oao_eco_freq',r.oao_eco_freq,'Frecuencia del eco retenido')
    +cjRowG('oao_eco_intensidad',r.oao_eco_intensidad,'Intensidad del eco retenido')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">lo oído sesga la voz futura (imitación por historia, NO copia); aprender es libre</div>`;}}
);
