window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'homeostasis',tit:'⚖️ Homeostasis',w:4,h:4,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('H_homeostasis',r.H_homeostasis,'Salud homeostática')+cjGauge(r.H_homeostasis,'#5fd38a')+cjSpark(bf.H_homeostasis,'#5fd38a')
    +cjRowG('x_interna',r.x_interna,'Variable interna regulada')
    +cjRowT('en_rango',si(r.en_rango),'¿La variable interna está dentro del rango viable?');}}
);
