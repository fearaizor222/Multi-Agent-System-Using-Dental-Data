const $ = id => document.getElementById(id);
let session = null, ready = false, busy = false;
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Chưa kết nối được ứng dụng.');
  return data;
}
function controls() { $('message').disabled = !ready || busy; $('send').disabled = !ready || busy; $('image').disabled = !ready || busy; }
function notice(text, error=false) { $('notice').textContent=text; $('notice').classList.toggle('error',error); }
function scroll() { $('conversation').scrollTop=$('conversation').scrollHeight; }
function addMessage(role, text, result, elapsed) {
  $('welcome').hidden=true;
  const turn=document.createElement('div');turn.className='turn '+role;
  const wrap=document.createElement('div');wrap.className='bubble';
  if(role==='assistant'){const author=document.createElement('div');author.className='author';author.textContent='DENTAL ASSISTANT';wrap.append(author);}
  const content=document.createElement('div');content.textContent=text;wrap.append(content);
  if(result){
    const details=document.createElement('details');details.className='details';
    const summary=document.createElement('summary');summary.textContent=`Chi tiết xử lý · ${elapsed}s`;details.append(summary);
    const panel=document.createElement('div');panel.className='detail-content';
    for(const step of result.trace || []){
      const row=document.createElement('div');row.className='trace-row';
      const name=document.createElement('span');name.textContent=step.step;
      const time=document.createElement('span');time.textContent=`${step.status} · ${(step.milliseconds/1000).toFixed(1)}s`;
      row.append(name,time);panel.append(row);
    }
    const json=document.createElement('pre');json.className='json';
    json.textContent=JSON.stringify({decision:result.decision,models:result.models,retrieval:result.knowledge?.internal_retrieval,a2_draft:result.knowledge?.draft,a3_verification:result.knowledge?.verification,synthesis:result.final_answer,vision:result.vision,error:result.error},null,2);
    panel.append(json);details.append(panel);wrap.append(details);
  }
  turn.append(wrap);$('conversation').append(turn);scroll();
}
async function refreshStatus(){
  try{
    const wasReady=ready;const state=await api('/api/status');ready=state.status==='ready';
    $('status').textContent=ready?'Model sẵn sàng':state.status==='error'?'Lỗi khởi động':'Đang nạp model';
    $('status').className='status '+(ready?'':state.status==='error'?'error':'loading');
    if(state.status==='error')notice(state.error,true);
    else if(ready && !wasReady && !busy)notice('Hỏi về kiến thức nha khoa hoặc thử tiếp một câu trong cùng hội thoại.');
    controls();
  }catch(error){ready=false;$('status').textContent='Mất kết nối';$('status').className='status error';notice('Ứng dụng chưa kết nối. Kiểm tra cửa sổ đang chạy web.',true);controls();}
}
async function newSession(){
  session=(await api('/api/session',{})).session_id;
}
$('new-chat').addEventListener('click',async()=>{
  if(busy)return;
  try{await newSession();document.querySelectorAll('.turn').forEach(x=>x.remove());$('welcome').hidden=false;$('message').value='';$('image').value='';notice('Đã bắt đầu hội thoại mới.');}catch(error){notice(error.message,true);}
});
document.querySelectorAll('.suggestions button').forEach(button=>button.addEventListener('click',()=>{
  if(!ready || busy)return;$('message').value=button.textContent;$('chat-form').requestSubmit();
}));
$('message').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('chat-form').requestSubmit();}});
$('chat-form').addEventListener('submit',async event=>{
  event.preventDefault();const file=$('image').files[0];const text=$('message').value.trim() || (file?'Mô tả hình ảnh nha khoa.':'');if(!text || busy || !ready)return;
  busy=true;controls();$('new-chat').disabled=true;$('progress').hidden=false;$('stage').textContent='Đang gửi câu hỏi';$('elapsed').textContent='0s';
  try{
    if(!session)await newSession();
    let payload={session_id:session,message:text};
    if(file){
      if(file.size>6*1024*1024)throw new Error('Chọn ảnh nhỏ hơn 6 MB.');
      const data=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=()=>reject(new Error('Không đọc được ảnh.'));r.readAsDataURL(file);});
      payload.image={data};
    }
    const accepted=await api(file?'/api/vision':'/api/chat',payload);
    addMessage('user',text+(file?'\n[Ảnh: '+file.name+']':''));$('message').value='';$('image').value='';notice(file?'Đang chạy agent ảnh. Kết quả ảnh thử nghiệm được hiển thị riêng, chưa được A3 kiểm chứng.':'Các agent đang xử lý. Bạn có thể xem thời gian và bước hiện tại bên trên.');
    while(true){
      const job=await api(`/api/jobs/${accepted.job_id}?session_id=${session}`);
      $('stage').textContent=job.stage;$('elapsed').textContent=job.elapsed_seconds+'s';
      if(job.status==='completed'){
        addMessage('assistant',job.result.display_message,job.result,job.elapsed_seconds);
        notice('Kết quả đã lưu: '+job.output,job.result.status==='error'||job.result.status==='blocked');break;
      }
      if(job.status==='failed')throw new Error(job.error);
      await new Promise(resolve=>setTimeout(resolve,1000));
    }
  }catch(error){notice(error.message,true);}
  finally{busy=false;controls();$('new-chat').disabled=false;$('progress').hidden=true;if(ready)$('message').focus();}
});
refreshStatus();setInterval(refreshStatus,4000);
