window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'balbuceo',tit:'🎙 Libertad expresiva (balbuceo)',w:4,h:6,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('g_freq',r.g_freq,'Frecuencia del gesto vocal')+cjBip(r.g_freq,'#e8b86d')
    +cjRowG('g_intensidad',r.g_intensidad,'Intensidad del gesto vocal')+cjBip(r.g_intensidad,'#6db6ff')
    +cjRowG('g_pausa',r.g_pausa,'Pausa entre gestos')+cjGauge(r.g_pausa,'#b58cff')
    +cjRowG('g_repeticion',r.g_repeticion,'Repetición del gesto')+cjGauge(r.g_repeticion,'#ff8c6b')
    +cjRowT('g_bucket',r.g_bucket,'Gesto elegido')+cjSpark(bf.g_freq,'#e8b86d');}}
);
