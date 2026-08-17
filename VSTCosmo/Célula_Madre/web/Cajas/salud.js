// Cada fila lleva la SIGLA (la que va al CSV) y su nombre descriptivo del glosario;
// el valor pasa por formatear() (fracción → %, y si no, valor + unidad + rango).
window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'salud',tit:'❤️ Salud del cierre',w:4,h:4,render:(b,r,bf)=>{b.innerHTML=
     cjRowG('OI',r.OI,'Organismicidad integrada')+cjGauge(r.OI,'#5fd38a')+cjSpark(bf.OI,'#5fd38a')
    +cjRowG('Lambda_Cos',r.Lambda_Cos,'Razón cosmosemiótica (salud del cierre)')
    +cjRowT('invariantes_ok',si(r.invariantes_ok),'Invariantes de viabilidad κ cumplidos')
    +cjRowG('A_sys_env',r.A_sys_env,'Acoplamiento con el entorno');}}
);
