window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'memoria',tit:'🧠 Memoria',w:4,h:6,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('mem_familiaridad',r.mem_familiaridad,'Familiaridad de lo que oye')+cjGauge(r.mem_familiaridad,'#6db6ff')
    +cjRowG('mem_novedad',r.mem_novedad,'Novedad de lo que oye')+cjGauge(r.mem_novedad,'#b58cff')
    +cjRowG('mem_recall',r.mem_recall,'Recuerdo evocado')+cjGauge(r.mem_recall,'#5fd38a')
    +cjRowT('mem_recall_tipo',r.mem_recall_tipo,'Tipo de recuerdo evocado')
    +cjRowG('mem_episodios_n',r.mem_episodios_n,'Episodios guardados')
    +cjRowG('mem_relacional_confianza',r.mem_relacional_confianza,'Confianza en la memoria relacional')+cjGauge(r.mem_relacional_confianza,'#e8b86d');}}
);
