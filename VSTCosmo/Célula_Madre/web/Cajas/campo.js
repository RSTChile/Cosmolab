window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'campo',tit:'🌀 Campo Φ / Soma',w:4,h:4,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('Omega',r.Omega,'Orientación del organismo')+cjSpark(bf.Omega,'#e8b86d')
    +cjRowG('gradiente',r.gradiente,'Pendiente del campo (la sorpresa)')+cjSpark(bf.gradiente,'#5fd38a')
    +cjRowG('omega_A_L',r.omega_A_L,'Percibido-vs-esperado del oído izquierdo')
    +cjRowG('omega_A_R',r.omega_A_R,'Percibido-vs-esperado del oído derecho');}}
);
