window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'metabolismo',tit:'⚡ Metabolismo',w:4,h:6,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('met_energia',r.met_energia,'Reserva de energía')+cjGauge(r.met_energia,'#e8b86d')+cjSpark(bf.met_energia,'#e8b86d')
    +cjRowG('met_hambre',r.met_hambre,'Hambre')+cjGauge(r.met_hambre,'#ff8c6b')
    +cjRowG('met_saciedad',r.met_saciedad,'Saciedad general')+cjGauge(r.met_saciedad,'#5fd38a')
    +cjRowG('met_nutricion',r.met_nutricion,'Nutrición del bocado')
    +cjRowG('met_balance',r.met_balance,'Balance: comió menos gastó')
    +cjRowT('met_clase',r.met_clase,'Veredicto: nutritiva / neutra / tóxica');}}
);
