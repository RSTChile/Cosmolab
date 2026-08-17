window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'voz',tit:'🔊 Voz / Comunicación',w:4,h:4,render:(b,r,bf)=>{b.innerHTML=
     cjRowT('voz_emitida',r.voz_emitida,'Sonido que está emitiendo')
    +cjRowG('voz_arousal',r.voz_arousal,'Activación de la voz')+cjGauge(r.voz_arousal,'#ff8c6b')
    +cjRowG('voz_valence',r.voz_valence,'Valencia de la voz (agrado)')+cjBip(r.voz_valence,'#6db6ff')
    +cjRowG('disposicion_cooperar',r.disposicion_cooperar,'Disposición a cooperar');}}
);
