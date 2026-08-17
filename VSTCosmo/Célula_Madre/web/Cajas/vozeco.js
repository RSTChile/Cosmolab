window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'vozeco',tit:'🌱 Valor ecológico de la voz',w:4,h:5,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('voz_otro_valor_ecologico',r.voz_otro_valor_ecologico,'Valor ecológico de la voz del otro')+cjGauge(Math.min(1,(+r.voz_otro_valor_ecologico||0)*8),'#8ef0c0')+cjSpark(bf.voz_otro_valor_ecologico,'#8ef0c0')
    +cjRowG('voz_otro_confianza_ecologica',r.voz_otro_confianza_ecologica,'Confianza en ese valor')+cjGauge(r.voz_otro_confianza_ecologica,'#e8b86d')
    +cjRowG('voz_otro_efecto_real',r.voz_otro_efecto_real,'Efecto real sobre su persistencia')+cjBip(Math.max(-1,Math.min(1,(+r.voz_otro_efecto_real||0)*4)),'#6db6ff')
    +cjRowG('voz_otro_modulacion_aplicada',r.voz_otro_modulacion_aplicada,'Modulación de absorción aplicada')
    +`<div class="obsk" style="font-size:9px;margin-top:3px">¿la voz del otro IMPORTA para persistir? cae bajo NULL/SHUFFLED</div>`;}}
);
