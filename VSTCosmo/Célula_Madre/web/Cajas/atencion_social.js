// Caja "Atención Social": muestra la decisión autónoma del organelo VST_OrganoAtencionSocial
// (a quién ESCUCHA y a quién HABLA, el sesgo dominante y los sesgos del objetivo). Columnas as_*.
// Cada fila lleva la sigla que va al CSV y, debajo, su nombre descriptivo del glosario.
window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push(
{id:'atencion_social', tit:'🧭 Atención Social', w:4, h:8,
 ayuda:'El organismo decide POR SÍ MISMO a quién ESCUCHAR y a quién HABLARLE entre los organismos '
  +'del Observatorio, sin que el usuario lo elija. Usa sesgos de atención social: ÉXITO (seguir a quien le va bien), '
  +'SIMILITUD (atender a los parecidos en género/tono/edad), DOMINANCIA (observar a los fuertes/activos), '
  +'y —cuando hay varios— PRESTIGIO (a quién atienden muchos) y CONFORMIDAD (lo que hace la mayoría). '
  +'Tiene LIBERTAD FUNCIONAL de no atender a nadie (más probable cuando su necesidad es alta) y compromiso '
  +'(no salta de interlocutor cada instante). Aquí ves: a quién escucha/habla, el sesgo dominante y los tres sesgos del objetivo.',
 render:(b,r,bf)=>{
  const nn = (x)=> (x===undefined||x===null||x===''||x==='-') ? '—' : x;
  const g  = (x)=> Number(x)||0;
  const sesgo = (r.as_esc_sesgo && r.as_esc_sesgo!=='-') ? ' · '+r.as_esc_sesgo : '';
  const sesgoH= (r.as_habla_sesgo && r.as_habla_sesgo!=='-') ? ' · '+r.as_habla_sesgo : '';
  const expl = Number(r.as_esc_explorando) ? ' · explorando' : '';
  b.innerHTML=
     cjRowT('as_modo', nn(r.as_modo),'Modo de atención social')
    +cjRowT('as_n_candidatos', nn(r.as_n_candidatos),'Candidatos en el campo')
    +cjRowT('as_esc_nombre', nn(r.as_esc_nombre)+sesgo,'👂 A quién escucha (y por qué sesgo)')
    +cjGauge(g(r.as_esc_score),'#f0c86e')+cjSpark(bf.as_esc_score,'#f0c86e')
    +cjRowT('as_habla_nombre', nn(r.as_habla_nombre)+sesgoH,'🗣 A quién habla (y por qué sesgo)')
    +cjRowT('as_esc_dwell_s', cjN(g(r.as_esc_dwell_s),1)+' s'+expl,'Tiempo que lleva con ese interlocutor')
    +'<div style="opacity:.65;font-size:.82em;margin:6px 0 2px">sesgos del objetivo</div>'
    +cjRowG('as_b_exito', r.as_b_exito,'Sesgo de éxito: seguir a quien le va bien')+cjGauge(g(r.as_b_exito),'#5fd38a')
    +cjRowG('as_b_similitud', r.as_b_similitud,'Sesgo de similitud: atender a los parecidos')+cjGauge(g(r.as_b_similitud),'#6db6ff')
    +cjRowG('as_b_dominancia', r.as_b_dominancia,'Sesgo de dominancia: observar a los fuertes')+cjGauge(g(r.as_b_dominancia),'#ff8c6d');
}}
);
