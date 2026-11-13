"""Local test website for the four real text agents. No cloud inference or training."""
import argparse
import base64
import binascii
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import uuid

from src.core.a1_process import ROOT
from src.graph.assistant_runtime import AssistantBundle

WEB = ROOT / "src/web"
STAGES = {"a1_plan":"A1 tiếp nhận yêu cầu", "a1_validate":"A2 bắt đầu tra tài liệu",
          "a2_retrieve":"A2 đang soạn bản nháp", "a2_draft":"A3 đang đối chiếu bằng chứng",
          "a3_verify":"Đã đối chiếu bằng chứng",
          "a2_a3":"A4 đang tổng hợp câu trả lời", "a4_synthesize":"Hoàn tất"}


class LocalHTTPServer(ThreadingHTTPServer):
    allow_reuse_address=False

    def server_bind(self):
        if os.name=="nt":
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


class WebApplication:
    def __init__(self, config, bundle=None):
        self.bundle = bundle if bundle is not None else AssistantBundle(config)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dental-inference")
        self.status = "loading"
        self.error = None
        self.identity = None
        self.sessions = {}
        self.jobs = {}
        self.executor.submit(self._warmup)

    def _warmup(self):
        try:
            identity = self.bundle.warmup()
            with self.lock:
                self.identity, self.status = identity, "ready"
        except Exception as exc:
            with self.lock:
                self.status, self.error = "error", type(exc).__name__ + ": " + str(exc)

    def public_status(self):
        with self.lock:
            return {"status":self.status,"error":self.error,"models":self.identity,
                    "active_jobs":sum(j["status"] in ("queued","running") for j in self.jobs.values()),
                    "image_agent": bool((self.identity or {}).get('image_agent'))}

    def new_session(self):
        with self.lock:
            if len(self.sessions)>=100:
                # Only remove an old idle session, never one with an active request.
                active={j["session_id"] for j in self.jobs.values() if j["status"] in ("queued","running")}
                old=next((key for key in self.sessions if key not in active),None)
                if old is None:raise ValueError("Quá nhiều hội thoại đang xử lý")
                del self.sessions[old]
                self.jobs={k:j for k,j in self.jobs.items() if j["session_id"]!=old}
            key=uuid.uuid4().hex
            self.sessions[key]={"history":[],"turn":0}
            return key

    def submit(self, session_id, message, image=None):
        if not isinstance(message,str) or not message.strip() or len(message)>2000:
            raise ValueError("Câu hỏi cần có từ 1 đến 2.000 ký tự")
        with self.lock:
            if self.status!="ready":raise ValueError("Model chưa sẵn sàng; xem trạng thái khởi động")
            if session_id not in self.sessions:raise ValueError("Hội thoại không tồn tại; tạo hội thoại mới")
            active=[j for j in self.jobs.values() if j["status"] in ("queued","running")]
            if any(j["session_id"]==session_id for j in active):raise ValueError("Hội thoại đang xử lý một câu hỏi")
            if len(active)>=4:raise ValueError("Đang có nhiều yêu cầu; vui lòng thử lại sau")
            # Keep bounded polling history; full results remain in outputs/web_sessions.
            finished=[k for k,j in self.jobs.items() if j["status"] in ("completed","failed")]
            for key in finished[:-50]:del self.jobs[key]
            key=uuid.uuid4().hex
            image_path = None
            if image is not None:
                if not (self.identity or {}).get('image_agent'):
                    raise ValueError('Agent ảnh chưa được bật')
                image_path = save_upload(image, key)
            self.jobs[key]={"id":key,"session_id":session_id,"status":"queued","stage":"Đang chờ xử lý",
                            "created":time.time(),"result":None}
            self.executor.submit(self._run_vision if image_path else self._run,key,message.strip(),
                                 *([image_path] if image_path else []))
            return key

    def _run_vision(self, job_id, message, image_path):
        with self.lock:
            job = self.jobs[job_id]
            job.update(status='running', stage='Agent ảnh đang phân tích', started=time.time())
        try:
            vision = self.bundle.analyze_image(image_path, message)
            findings = vision['findings']
            text = 'Phân tích ảnh thử nghiệm — chưa được A3 kiểm chứng.\n\n' + findings['preliminary_impression']
            for item in findings['detected_pathologies']:
                text += '\n• ' + item['name'] + ' — ' + item['location'] + ': ' + item['visual_evidence']
            result = {'status': 'image_analysis', 'display_message': text, 'vision': vision,
                      'models': self.bundle.client.identity, 'trace': [], 'input': {'message': message, 'image': str(image_path)}}
            folder = ROOT / 'outputs/web_sessions' / job['session_id']
            folder.mkdir(parents=True, exist_ok=True)
            output = folder / ('image_' + job_id + '.json')
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            with self.lock:
                job.update(status='completed', stage='Hoàn tất', finished=time.time(), result=result,
                           output=str(output.relative_to(ROOT)))
        except Exception as exc:
            with self.lock:
                job.update(status='failed', stage='Phân tích ảnh chưa hoàn tất', finished=time.time(),
                           error=type(exc).__name__ + ': ' + str(exc))

    def _run(self, job_id, message):
        with self.lock:
            job=self.jobs[job_id];session=self.sessions[job["session_id"]]
            history=list(session["history"])
            job.update(status="running",stage="A1 tiếp nhận yêu cầu",started=time.time())
        def progress(node,status):
            with self.lock:
                job["stage"]=STAGES.get(node,node)
        self.bundle.on_progress=progress
        try:
            result={"input":{"message":message,"history":history}}
            for event in self.bundle.graph.stream(result,stream_mode="updates"):
                for node,update in event.items():
                    result.update(update)
                    progress(node,update.get("status"))
            result["run_at"]=datetime.now(timezone.utc).isoformat()
            result["models"]=self.bundle.client.identity
            result["display_message"]=result.get("message") or "Chưa có câu trả lời. Xem chi tiết kết quả xử lý."
            directory=ROOT/"outputs/web_sessions"/job["session_id"]
            directory.mkdir(parents=True,exist_ok=True)
            session["turn"]+=1
            output=directory/f'turn_{session["turn"]:03d}.json'
            output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            with self.lock:
                if result.get("status") not in ("error","blocked"):
                    # Explicitly limited conversation context, displayed as such in the UI.
                    session["history"]=(history+[{"role":"user","content":message},
                        {"role":"assistant","content":result["display_message"]}])[-12:]
                job.update(status="completed",stage="Hoàn tất",finished=time.time(),result=result,
                           output=str(output.relative_to(ROOT)))
        except Exception as exc:
            with self.lock:
                job.update(status="failed",stage="Xử lý chưa hoàn tất",finished=time.time(),
                           error=type(exc).__name__+": "+str(exc))
        finally:
            self.bundle.on_progress=None

    def job(self, job_id, session_id):
        with self.lock:
            job=self.jobs.get(job_id)
            if job is None or job["session_id"]!=session_id:raise KeyError("Không tìm thấy lượt xử lý")
            return {**job,"elapsed_seconds":round((job.get("finished",time.time())-job.get("started",job["created"])),1)}

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
        self.bundle.close()


def save_upload(payload, key):
    from PIL import Image, ImageOps
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), str):
        raise ValueError('Dữ liệu ảnh không hợp lệ')
    try:
        data = base64.b64decode(payload['data'], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('Mã hóa ảnh không hợp lệ') from exc
    if not 0 < len(data) <= 6 * 1024**2:
        raise ValueError('Ảnh cần nhỏ hơn 6 MB')
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in ('JPEG', 'PNG', 'WEBP') or source.width * source.height > 20_000_000:
                raise ValueError('Chỉ nhận ảnh JPG, PNG, WEBP, tối đa 20 triệu điểm ảnh')
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((1600, 1600))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Không đọc được ảnh') from exc
    folder = ROOT / 'outputs/web_uploads'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (key + '.png')
    image.save(path)
    return path


def handler_for(application):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data, content_type="application/json; charset=utf-8"):
            body=json.dumps(data,ensure_ascii=False).encode("utf-8") if content_type.startswith("application/json") else data
            self.send_response(code)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)

        def local_request(self):
            host=self.headers.get("Host","")
            try:
                parsed=urlsplit("http://"+host)
                if parsed.hostname not in ("127.0.0.1","localhost") or parsed.port!=self.server.server_port:
                    return False
            except ValueError:return False
            origin=self.headers.get("Origin")
            return origin is None or origin=="http://"+host

        def do_GET(self):
            if not self.local_request():self.reply(403,{"error":"Chỉ nhận yêu cầu từ trang web cục bộ"});return
            from urllib.parse import parse_qs
            url=urlsplit(self.path)
            assets={"/":("index.html","text/html; charset=utf-8"),"/app.js":("app.js","text/javascript; charset=utf-8"),
                    "/style.css":("style.css","text/css; charset=utf-8")}
            if url.path in assets:
                file,kind=assets[url.path];self.reply(200,(WEB/file).read_bytes(),kind)
            elif url.path=="/favicon.ico":self.reply(200,b"","image/x-icon")
            elif url.path=="/api/status":self.reply(200,application.public_status())
            elif url.path.startswith("/api/jobs/"):
                try:self.reply(200,application.job(url.path.split("/")[-1],parse_qs(url.query).get("session_id",[""])[0]))
                except KeyError:self.reply(404,{"error":"Không tìm thấy lượt xử lý"})
            else:self.reply(404,{"error":"Không tìm thấy trang"})

        def do_POST(self):
            if not self.local_request():self.reply(403,{"error":"Chỉ nhận yêu cầu từ trang web cục bộ"});return
            if self.headers.get("Content-Type","").split(";")[0]!="application/json":
                self.reply(415,{"error":"Cần gửi JSON"});return
            try:
                length=int(self.headers.get("Content-Length","0"))
                limit = 9 * 1024**2 if self.path == '/api/vision' else 16000
                if not 0<length<=limit:raise ValueError("Dữ liệu gửi lên không hợp lệ")
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ValueError("Cần một JSON object")
                if self.path=="/api/session":self.reply(201,{"session_id":application.new_session()})
                elif self.path=="/api/chat":
                    key=application.submit(payload.get("session_id"),payload.get("message"))
                    self.reply(202,{"job_id":key})
                elif self.path == '/api/vision':
                    if 'image' not in payload:
                        raise ValueError('Cần chọn ảnh')
                    key = application.submit(payload.get('session_id'), payload.get('message') or 'Mô tả hình ảnh nha khoa.', payload['image'])
                    self.reply(202, {'job_id': key})
                else:self.reply(404,{"error":"Không tìm thấy API"})
            except (ValueError,TypeError,KeyError) as exc:self.reply(400,{"error":str(exc)})

        def log_message(self, fmt, *args):
            # Polling should not flood the console or log question contents.
            if args and str(args[1]) not in ("200","201","202"):super().log_message(fmt,*args)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/assistant_local.json")
    parser.add_argument("--port",type=int,default=8000)
    args=parser.parse_args()
    try:
        server=LocalHTTPServer(("127.0.0.1",args.port),BaseHTTPRequestHandler)
    except OSError:
        parser.exit(1,f'Cổng {args.port} đang được dùng. Mở http://127.0.0.1:{args.port} hoặc chọn --port khác.\n')
    app=None
    try:
        app=WebApplication(args.config)
        server.RequestHandlerClass=handler_for(app)
        print(f'Web thử nghiệm: http://127.0.0.1:{args.port}',flush=True)
        print("Đang nạp Qwen3 và các kho RAG. Mở web để theo dõi trạng thái. Ctrl+C để dừng.",flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Đang đóng ứng dụng...",flush=True)
    finally:
        server.server_close()
        if app is not None:app.close()


if __name__=="__main__":
    main()
