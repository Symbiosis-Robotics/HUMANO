import content from './content.json';
import registry from '../../resources/data/task_metadata.json';
import taskDemos from '../../resources/data/task_demos.json';
import results from '../../resources/data/benchmark_results.json';

const D = {
  ...content,
  tasks: registry.tasks,
  taskDemos,
  idResults: results.idResults,
  oodResults: results.oodResults
};
const $=(q,root=document)=>root.querySelector(q);
const esc=(v)=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths={
 'file-text':'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h8"/>',
 'code-xml':'<path d="m18 16 4-4-4-4"/><path d="m6 8-4 4 4 4"/><path d="m14.5 4-5 16"/>',
 'database':'<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
 'play':'<polygon points="6 3 20 12 6 21 6 3"/>',
 'image':'<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>',
 'hand':'<path d="M8 12V5a1.5 1.5 0 0 1 3 0v6"/><path d="M11 7a1.5 1.5 0 0 1 3 0v4"/><path d="M14 8a1.5 1.5 0 0 1 3 0v4"/><path d="M17 10a1.5 1.5 0 0 1 3 0v6a6 6 0 0 1-6 6h-2a7 7 0 0 1-6-4l-3-5a1.5 1.5 0 0 1 2.5-1.5L8 15"/>',
 'package':'<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="M3 7v10l9 5 9-5V7"/><path d="M12 12v10"/><path d="m7.5 4.5 9 5"/>',
 'move':'<path d="m5 9-3 3 3 3"/><path d="m9 5 3-3 3 3"/><path d="m15 19-3 3-3-3"/><path d="m19 9 3 3-3 3"/><path d="M2 12h20"/><path d="M12 2v20"/>',
 'shapes':'<rect x="3" y="3" width="8" height="8" rx="1"/><circle cx="17" cy="7" r="4"/><path d="m7 14-4 7h8z"/><path d="m17 13-4 8h8z"/>',
 'footprints':'<path d="M4 16v3a2 2 0 0 0 4 0v-3"/><path d="M3 12c0-4 2-8 4-8 2 0 2 3 2 6v5H3z"/><path d="M15 18v2a2 2 0 0 0 4 0v-2"/><path d="M14 14c0-4 2-8 4-8 2 0 2 3 2 6v5h-6z"/>',
 'target':'<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
 'list-ordered':'<path d="M10 6h11M10 12h11M10 18h11"/><path d="M4 6h1v4M4 10h2M4 16h2l-2 3h2"/>',
 'git-branch':'<line x1="6" y1="3" x2="6" y2="15"/><circle cx="6" cy="3" r="2"/><circle cx="6" cy="17" r="2"/><circle cx="18" cy="6" r="2"/><path d="M18 8a6 6 0 0 1-6 6H6"/>',
 'network':'<rect x="2" y="8" width="6" height="6" rx="1"/><rect x="16" y="2" width="6" height="6" rx="1"/><rect x="16" y="16" width="6" height="6" rx="1"/><path d="M8 11h4a7 7 0 0 0 7-7M8 11h4a7 7 0 0 1 7 7"/>',
 'scan-eye':'<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M2 12s3.5-5 10-5 10 5 10 5-3.5 5-10 5S2 12 2 12Z"/><circle cx="12" cy="12" r="2"/>',
 'workflow':'<rect x="3" y="3" width="8" height="8" rx="2"/><rect x="13" y="13" width="8" height="8" rx="2"/><path d="M7 11v6a2 2 0 0 0 2 2h4"/><path d="M11 7h6a2 2 0 0 1 2 2v4"/>',
 'person-standing':'<circle cx="12" cy="4" r="2"/><path d="M12 6v7m-6 7 6-7 6 7M6 8l6-2 6 2"/>',
 'waypoints':'<circle cx="12" cy="4.5" r="2.5"/><path d="m10.2 6.3-4.4 4.4M14 6.5l4.3 4.3"/><circle cx="4.5" cy="13.5" r="2.5"/><circle cx="19.5" cy="13.5" r="2.5"/><path d="M6.9 15h10.2"/><path d="M12 15v6"/>',
 'bot':'<rect x="3" y="7" width="18" height="14" rx="3"/><path d="M12 7V3M8 3h8"/><circle cx="8.5" cy="13" r="1"/><circle cx="15.5" cy="13" r="1"/><path d="M8 17h8"/>',
 'monitor-play':'<rect x="2" y="3" width="20" height="15" rx="2"/><path d="M8 22h8M12 18v4"/><path d="m10 7 5 4-5 4V7z"/>',
 'search':'<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>',
 'copy':'<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>'
};
const icon=(name,cl='')=>`<svg class="${cl}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]||paths.image}</svg>`;
function decorateIcons(scope=document){scope.querySelectorAll('[data-icon]').forEach(el=>{el.innerHTML=icon(el.dataset.icon)})}
function makeResources(node) {
  if (!node) return;
  const links = [
    ['Paper', 'paper'],
    ['Code', 'code', 'github']
  ];
  node.innerHTML = links.map(([label, key, brand]) => {
    const url = D.resourceLinks[key];
    const mark = brand ? `<img src="icons/${brand}.svg" width="18" height="18" alt="">` : icon('file-text');
    const body = `${mark}<span>${label}</span>`;
    return url
      ? `<a class="resource-button" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${body}</a>`
      : `<span class="resource-button" aria-disabled="true">${body}</span>`;
  }).join('');

  const mirrors = [['ModelScope', 'modelscope'], ['Hugging Face', 'huggingface']];
  const choices = mirrors.map(([label, key]) =>
    `<a href="${esc(D.resourceLinks[key])}" target="_blank" rel="noopener noreferrer"><img src="icons/${key}.svg" width="18" height="18" alt="">${label}</a>`
  ).join('');
  node.insertAdjacentHTML('beforeend', `<details class="dataset-picker">
    <summary class="resource-button">${icon('database')} Data <span aria-hidden="true">▾</span></summary>
    <div class="dataset-mirrors">${choices}</div>
  </details>`);
  const picker = node.querySelector('.dataset-picker');
  document.addEventListener('click', event => {
    if (!picker.contains(event.target) || event.target.closest('.dataset-mirrors a')) picker.open = false;
  });
  picker.addEventListener('keydown', event => {
    if (event.key === 'Escape' && picker.open) {
      picker.open = false;
      picker.querySelector('summary').focus();
    }
  });
}
function media(cfg, text, {light=false,autoplay=false}={}){
 if(cfg?.video)return `<video ${autoplay?'autoplay muted loop':'controls'} playsinline preload="metadata" ${cfg.poster?'poster="'+esc(cfg.poster)+'"':''} aria-label="${esc(text)}"><source src="${esc(cfg.video)}" type="video/mp4">Your browser does not support video.</video>`;
 if(cfg?.image)return `<img src="${esc(cfg.image)}" loading="lazy" alt="${esc(text)}">`;
 return `<div class="media-slot${light?' media-slot--light':''}">${icon('play','slot-icon')}<span class="slot-caption">${esc(text)}</span></div>`
}
function setupNav(){const button=$('#mobile-menu');if(button)button.addEventListener('click',()=>{const nav=$('.main-nav');const opened=nav.classList.toggle('is-open');button.setAttribute('aria-expanded',String(opened));button.setAttribute('aria-label',opened?'Close navigation':'Open navigation')})}
function hero(){const box=$('#hero-mosaic');if(!box)return;box.innerHTML=D.hero.map(t=>`<div class="hero-tile ${t.video?'':'is-empty'}">${media({video:t.video,poster:t.poster},t.video?t.title+' demonstration':'DEMONSTRATION PENDING',{autoplay:true})}<div class="hero-tile-bottom"><div><small>${esc(t.suite)}</small><br><b>${esc(t.title)}</b></div><span class="task-id">${esc(t.id)}</span></div></div>`).join('')}
function illustration(group,name,extension){return 'resources/illustrations/'+group+'/'+name.toLowerCase().replaceAll(' ','-')+'.'+extension}
function suites(){const box=$('#suite-grid');if(!box)return;box.innerHTML=D.suites.map(s=>`<a class="suite-card" href="tasks.html?suite=${encodeURIComponent(s.name)}"><span class="suite-no">${esc(s.no)} / CAPABILITY SUITE</span><img class="suite-art" src="${illustration('task-suites',s.name,'png')}" alt="" loading="lazy"><div><strong>${esc(s.name)}</strong><p>${esc(s.summary)}</p><span class="suite-count">${s.count} TASKS ↗</span></div></a>`).join('')}
let currentOOD=0;
function oodPanel(i){const f=D.ood[i];currentOOD=i;const buttons=$('#ood-tabs');if(!buttons)return;buttons.querySelectorAll('button').forEach((el,j)=>{el.classList.toggle('is-active',j===i);el.setAttribute('aria-selected',String(j===i))});$('#ood-number').textContent=String(i+1).padStart(2,'0')+' / 07';$('#ood-factor-title').textContent=f.name;$('#ood-illustration').src=illustration('ood-settings',f.name,'png');$('#ood-camera-note').textContent=f.note;
 const pair=$('#ood-pair');
 const first=f.options?.[0]||{};
 const card=(id)=>`<div class="comparison-card"><div class="comparison-top"><span>${id?'IN-DISTRIBUTION':'OUT-OF-DISTRIBUTION'}</span><span>${id?'01':'02'}</span></div><div class="ood-sample">${oodSample(f,first,id)}</div><div class="comparison-foot">${esc(f.task||'Representative task to be confirmed')}</div></div>`;
 const choice=$('#ood-option'), select=choice.querySelector('select');
 pair.innerHTML=card(true)+card(false);
 // Camera view changes how both initial poses are observed.
 if(f.key==='robot') pair.before(choice);
 else pair.lastElementChild.append(choice);
 choice.hidden=!f.options;
 if(f.options){
  choice.querySelector('span').textContent=f.key==='robot'?'Camera view':f.key==='language'?'Instruction':f.key==='background'?'Background':'Condition';
  select.innerHTML=f.options.map((o,n)=>`<option value="${n}">${esc(o.label)}</option>`).join('');
 }
 renderOODFacts(f,first);
}
function oodSample(f,option,id){
 const selected={...f.media,...option};
 if(f.mode==='language'){
  const sample=id?selected.idText:selected.oodText;
  return `<div class="text-comparison"><strong>${id?'CANONICAL INSTRUCTION':'HELD-OUT INSTRUCTION'}</strong><p>${esc(sample||'[Verified instruction text pending]')}</p></div>`;
 }
 const src=id?selected.id:selected.ood;
 return media(src?(f.media.type==='video'?{video:src}:{image:src}):{},`${f.task} · ${f.name} · ${id?'ID':'OOD'}`,{light:true});
}
function renderOODFacts(f,option){
 $('#ood-facts').innerHTML=[['WHAT CHANGES',option.change||f.change],['WHAT STAYS FIXED',f.fixed],['CAPABILITY EVALUATED',f.evaluates]].map(([heading,desc])=>`<div><span>${heading}</span><p>${esc(desc)}</p></div>`).join('');
}
function ood(){
 const box=$('#ood-tabs');if(!box)return;
 box.innerHTML=D.ood.map((x,i)=>`<button type="button" role="tab" aria-selected="${i===0}" data-ood="${i}"><img src="${illustration('ood-settings',x.name,'svg')}" alt=""><span>${esc(x.name)}</span><small>${String(i+1).padStart(2,'0')}</small></button>`).join('');
 box.addEventListener('click',e=>{const btn=e.target.closest('[data-ood]');if(btn)oodPanel(Number(btn.dataset.ood))});
 $('#ood-option select').addEventListener('change',e=>{
  const f=D.ood[currentOOD], option=f.options[Number(e.target.value)];
  const [left,right]=document.querySelectorAll('#ood-pair .ood-sample');
  right.innerHTML=oodSample(f,option,false);
  if(f.key==='robot')left.innerHTML=oodSample(f,option,true);
  renderOODFacts(f,option);
 });
 oodPanel(0);
}
function leaderboard(){const node=$('#leaderboard-table');if(!node)return;const render=mode=>{const source=mode==='id'?D.idResults:D.oodResults;const maxScores=source.columns.map((_,i)=>Math.max(...source.rows.map(r=>r.scores[i])));const highestAvg=Math.max(...source.rows.map(r=>r.reportedAverage));const sorted=[...source.rows].sort((a,b)=>b.reportedAverage-a.reportedAverage);node.innerHTML=`<thead><tr><th scope="col">MODEL</th>${source.columns.map(c=>`<th scope="col">${esc(c)}</th>`).join('')}<th scope="col" class="overall-col">${mode==='id'?'REPORTED AVG.':'OOD AVG.'}</th></tr></thead><tbody>${sorted.map(r=>`<tr><td>${esc(r.model)}</td>${r.scores.map((score,j)=>`<td class="${score===maxScores[j]?'best':''}">${score.toFixed(1)}</td>`).join('')}<td class="overall-col ${r.reportedAverage===highestAvg?'best':''}">${r.reportedAverage.toFixed(1)}</td></tr>`).join('')}</tbody>`;
 document.querySelectorAll('[data-results-tab]').forEach(b=>{const active=b.dataset.resultsTab===mode;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))})};
 document.querySelectorAll('[data-results-tab]').forEach(b=>b.addEventListener('click',()=>render(b.dataset.resultsTab)));render('id')}
const actionSequences={
  'L-6':['Pick up the box from the ground.','Place the box on the dolly.','Push the dolly forward.'],
  'L-10':['Pick up the bottle from the ground.','Place the bottle inside the box on the table.','Lift the box.']
};
function episodeOptions(episodes,selected=episodes[0]){
 return episodes.map(ep=>`<option value="${ep}" ${ep===selected?'selected':''}>Episode ${ep}</option>`).join('');
}
function demoVariants(task,demos){
 if(!task.variants)return [];
 const mapped=new Set(task.variants.flatMap(v=>v.episodes));
 const archived=demos.filter(d=>!mapped.has(d.episode)).map(d=>({
   label:'Archived demonstration · Episode '+d.episode,episodes:[d.episode]
 }));
 return [...task.variants,...archived];
}
function initialDemo(task){
 const demos=D.taskDemos[task.id]||[];
 const mapped=new Set((task.variants||[]).flatMap(v=>v.episodes));
 return demos.find(d=>mapped.has(d.episode))||demos[0];
}
function demoPanel(task,first){
 const demos=D.taskDemos[task.id]||[];
 if(!demos.length)return media({},'Demonstration pending',{light:true});
 const variants=demoVariants(task,demos);
 const selected=variants.findIndex(v=>v.episodes.includes(first.episode));
 const branchControl=variants.length?`<label>Variant <select data-demo-branch>${variants.map((v,i)=>`<option value="${i}" ${i===selected?'selected':''}>${esc(v.label)}</option>`).join('')}</select></label>`:'';
 const variant=variants[selected];
 const episodes=variant?.episodes||[first.episode];
 const episodeControl=variants.some(v=>v.episodes.length>1)?`<label data-episode-label ${episodes.length<2?'hidden':''}>Demonstration <select data-demo-episode>${episodeOptions(episodes,first.episode)}</select></label>`:'';
 return `<div class="demo-controls">${branchControl}${episodeControl}<label>Camera view <select data-demo-view><option value="firstPerson">Egocentric</option><option value="thirdPerson">Exocentric</option></select></label></div>`+
   `<div data-variant-description>${esc(variant?.description||'')}</div><div data-demo-player>${media({video:first.firstPerson},task.id+' egocentric demonstration')}</div>`;
}
function changeDemo(event){
 const control=event.target;
 if(!control.matches('[data-demo-branch],[data-demo-view],[data-demo-episode]'))return;
 const panel=control.closest('[data-demo-task]');
 const task=D.tasks.find(t=>t.id===panel.dataset.demoTask);
 const demos=D.taskDemos[task.id];
 const variant=demoVariants(task,demos)[Number(panel.querySelector('[data-demo-branch]')?.value)];
 const episode=panel.querySelector('[data-demo-episode]');
 if(episode && control.matches('[data-demo-branch]')){
   episode.innerHTML=episodeOptions(variant.episodes);
   episode.parentElement.hidden=variant.episodes.length<2;
 }
 const demo=variant?demos.find(d=>d.episode===(episode?.value?Number(episode.value):variant.episodes[0])):demos[0];
 const camera=panel.querySelector('[data-demo-view]');
 camera.disabled=!demo;
 panel.querySelector('[data-variant-description]').textContent=variant?.description||'';
 panel.closest('#task-detail').querySelector('[data-task-instruction]').textContent=(demo?demo.instruction:variant?.instruction)||task.instruction;
 const player=panel.querySelector('[data-demo-player]');
 if(!demo){player.textContent='No video for this variant.';return}
 const view=camera.value;
 const label=task.id+' '+(variant?.label||demo.label)+' '+(view==='firstPerson'?'egocentric':'exocentric')+' demonstration';
 const video=player.querySelector('video');
 if(!video){player.innerHTML=media({video:demo[view]},label);return}
 const position=control===camera?video.currentTime:0;
 const playing=!video.paused;
 video.onloadedmetadata=()=>{
   video.currentTime=Math.min(position,video.duration);
   if(playing)video.play().catch(()=>{});
 };
 video.setAttribute('aria-label',label);
 video.src=demo[view];
}
function taskCode(task){
 return esc(task.environmentId.replace(/-v0$/, ''));
}
function taskDetail(task){const first=initialDemo(task);let custom='';if(actionSequences[task.id])custom=`<div class="detail-section"><h3>ACTION SEQUENCE (UNSEGMENTED)</h3><div class="task-step-seq">${actionSequences[task.id].map((s,i)=>`<div class="task-step"><b>${String(i+1).padStart(2,'0')}</b><span>${esc(s)}</span></div>`).join('')}</div></div>`;
 const infoLabel=task.suite==='Goal' ? (['G-5','G-9'].includes(task.id)?'TARGET OBJECTS':'DESTINATIONS') :
   ({Spatial:'SPATIAL VARIATION',Object:'TARGET OBJECTS',Conditional:'INITIAL CONDITIONS'}[task.suite] || 'SKILL');
 return `<div class="detail-header"><span class="kicker-small detail-suite"><img src="${illustration('task-suites',task.suite,'svg')}" alt="">${esc(task.suite)} / ${esc(task.id)}</span><span class="kicker-small">CANONICAL TASK</span></div><h2 class="task-code">${taskCode(task)}</h2><p class="task-description">${esc(task.title)}</p><div class="detail-media" data-demo-task="${esc(task.id)}">${demoPanel(task,first)}</div><div class="detail-section"><h3>TASK INSTRUCTION</h3><p data-task-instruction>${esc(first?.instruction||task.instruction)}</p></div>${custom}<div class="detail-section"><h3>${infoLabel}</h3><p>${esc(task.annotations)}</p></div><div class="detail-metadata"><div><strong>${task.demos}</strong><span>DEMOS PER TASK</span></div><div><strong>${esc(task.suite)}</strong><span>TASK SUITE</span></div></div>`
}
function explorer(){const list=$('#task-list'),detail=$('#task-detail');if(!list||!detail)return;const select=$('#task-suite-filter'),search=$('#task-search'),counter=$('#task-counter');const url=new URLSearchParams(location.search);const initial=url.get('suite');const names=['All Suites',...D.suites.map(s=>s.name)];select.innerHTML=names.map(n=>`<option value="${esc(n)}">${esc(n)}</option>`).join('');select.value=names.includes(initial)?initial:'All Suites';let selected='B-9';let filtered=[];
 function render(){const q=search.value.trim().toLowerCase();filtered=D.tasks.filter(t=>(select.value==='All Suites'||t.suite===select.value)&&(!q||[t.id,t.suite,t.title,t.instruction,...(t.alternateInstructions||[]),t.environmentId,t.annotations].some(v=>String(v).toLowerCase().includes(q))));counter.textContent=`${filtered.length} / 100 tasks`;list.innerHTML=filtered.length?filtered.map(t=>`<button class="task-item ${t.id===selected?'is-selected':''}" type="button" data-task="${esc(t.id)}"><span class="item-id">${esc(t.id)} · ${esc(t.suite)}</span><strong class="task-code">${taskCode(t)}</strong><span class="task-description">${esc(t.title)}</span><small>${esc(t.annotations)}</small></button>`).join(''):'<p class="no-results">No matching tasks. Try another term or suite.</p>';if(filtered.length&&!filtered.some(t=>t.id===selected))selected=filtered[0].id;if(!filtered.length){detail.innerHTML='<p class="no-results">Select a different filter to view task details.</p>';return}detail.innerHTML=taskDetail(D.tasks.find(t=>t.id===selected));detail.scrollTop=0;list.querySelectorAll('[data-task]').forEach(b=>b.classList.toggle('is-selected',b.dataset.task===selected))}
 detail.addEventListener('change',changeDemo);list.addEventListener('click',e=>{const el=e.target.closest('[data-task]');if(!el)return;selected=el.dataset.task;render();if(window.innerWidth<850)detail.scrollIntoView({behavior:'smooth',block:'start'})});search.addEventListener('input',render);select.addEventListener('change',render);render()}
setupNav();decorateIcons();makeResources($('#hero-resources'));hero();suites();ood();leaderboard();explorer();
