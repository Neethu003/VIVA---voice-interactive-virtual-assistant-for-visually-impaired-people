from flask import Flask, render_template, Response, request, jsonify
import threading, time, os, random, cv2
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import winsound
import torch

# ---------------- CONFIG ----------------
CAMERA_INDEX = 0
CUSTOM_MODEL_PATH = r"C:\Users\Neetu\Virtual Impaired\runs\train\blindassist_yolo_retry\weights\best.pt"
RESIZE_W, RESIZE_H = 480, 360
TOO_CLOSE_DISTANCE = 3.0  # meters
SPEECH_GAP = 5.0
POSITION_THRESHOLD = 0.3  # meters
TTS_TEMP_DIR = "."

# ---------------- FLAGS ----------------
detection_enabled = True
current_language = "en"

# ---------------- TTS ----------------
tts_lock = threading.Lock()
tts_cache = {}
gtts_codes = {"en":"en","hi":"hi","kn":"kn","te":"te","ta":"ta"}

def speak(text, lang="en"):
    def _job():
        with tts_lock:
            try:
                if text not in tts_cache:
                    fname = os.path.join(TTS_TEMP_DIR, f"tts_{random.randint(1000,9999)}.mp3")
                    gTTS(text=text, lang=lang).save(fname)
                    tts_cache[text] = fname
                playsound(tts_cache[text])
            except Exception as e:
                print("TTS error:", e)
    threading.Thread(target=_job, daemon=True).start()

# ---------------- TRANSLATIONS ----------------
direction_translate = {
    "en": {"left":"left","right":"right","center":"center"},
    "hi": {"left":"बाएँ","right":"दाएँ","center":"बीच"},
    "kn": {"left":"ಎಡ","right":"ಬಲ","center":"ಮಧ್ಯ"},
    "te": {"left":"ఎడమ","right":"కుడి","center":"మధ్య"},
    "ta": {"left":"இடது","right":"வலது","center":"நடு"},
}

warning_text = {
    "en":"Warning, object very close",
    "hi":"चेतावनी, वस्तु बहुत पास है",
    "kn":"ಎಚ್ಚರಿಕೆ, ವಸ್ತು ತುಂಬಾ ಹತ್ತಿರ ಇದೆ",
    "te":"హెచ్చరిక, వస్తువు చాలా దగ్గరలో ఉంది",
    "ta":"எச்சரிக்கை, பொருள் மிகவும் அருகில் உள்ளது"
}

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ---------------- LOAD MODELS ----------------
print("Loading YOLO models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
general_model = YOLO("yolov8n.pt")
custom_model = YOLO(CUSTOM_MODEL_PATH)
general_model.to(device)
custom_model.to(device)
print("✔ Models loaded")

# ---------------- HELPERS ----------------
def estimate_distance(h,H=1.6,f=600): 
    return round((H*f)/h,2) if h>0 else 0

def get_direction(cx,w=RESIZE_W):
    if cx<w*0.33: return "left"
    if cx>w*0.66: return "right"
    return "center"

def iou_box(a,b):
    xA=max(a[0],b[0]); yA=max(a[1],b[1]); xB=min(a[2],b[2]); yB=min(a[3],b[3])
    interW=max(0,xB-xA); interH=max(0,yB-yA); interArea=interW*interH
    if interArea==0: return 0.0
    areaA=(a[2]-a[0])*(a[3]-a[1])
    areaB=(b[2]-b[0])*(b[3]-b[1])
    return interArea/float(areaA+areaB-interArea)

def nms_same_class(boxes):
    out=[]; by_class={}
    for b in boxes: by_class.setdefault(b['cls'],[]).append(b)
    for cls,blist in by_class.items():
        blist=sorted(blist,key=lambda x:x['conf'],reverse=True)
        keep=[]
        while blist:
            cur=blist.pop(0)
            keep.append(cur)
            blist=[o for o in blist if iou_box(cur['box'],o['box'])<=0.5]
        out.extend(keep)
    return out

# ---------------- DETECTOR THREAD ----------------
class Detector(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cap=cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name=='nt' else 0)
        self.cap.set(3,640); self.cap.set(4,480)
        self.frame=None
        self.last_detections=[]
        self.running=True
        self.voice_timer=time.time()
        self.announced_objects={}
        self.detection_enabled=True
        self.current_language="en"

    def run(self):
        prev_any_close=False
        while self.running:
            ret,frame=self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.frame=frame.copy()
            detections=[]
            if self.detection_enabled:
                small=cv2.resize(frame,(RESIZE_W,RESIZE_H))
                try:
                    res_g=general_model(small,verbose=False)[0]
                    res_c=custom_model(small,verbose=False)[0]
                except Exception as e: 
                    print("Model inference error:",e)
                    continue
                for box,mdl in [(b,"general") for b in res_g.boxes]+[(b,"custom") for b in res_c.boxes]:
                    cls_idx=int(box.cls); conf=float(box.conf)
                    x1,y1,x2,y2=map(int,box.xyxy[0])
                    h=y2-y1; dist=estimate_distance(h)
                    direction=get_direction((x1+x2)//2)
                    name=COCO_NAMES[cls_idx] if mdl=="general" and cls_idx<len(COCO_NAMES) else custom_model.names.get(cls_idx,f"class_{cls_idx}")
                    detections.append({"cls":cls_idx,"name":name,"conf":conf,"box":[x1,y1,x2,y2],"dist":dist,"direction":direction})
                detections=nms_same_class(detections)
                self.last_detections=detections

                # Closest object for beeping
                closest_distance_cm=None
                any_close=False
                for d in detections:
                    if 0<d["dist"]<=TOO_CLOSE_DISTANCE: any_close=True
                    cm=d["dist"]*100
                    if closest_distance_cm is None or cm<closest_distance_cm: closest_distance_cm=cm
                # Multilevel beeping
                if closest_distance_cm is not None and closest_distance_cm<=TOO_CLOSE_DISTANCE*100:
                    if closest_distance_cm<75: winsound.Beep(1400,200)
                    elif closest_distance_cm<150: winsound.Beep(1200,150)
                    else: winsound.Beep(1000,100)

                # Warning TTS
                if any_close and not prev_any_close:
                    try: winsound.Beep(800,400)
                    except: pass
                    speak(warning_text[self.current_language], gtts_codes.get(self.current_language,"en"))
                prev_any_close=any_close

                # Object announcement
                now=time.time()
                if detections and (now-self.voice_timer)>=SPEECH_GAP:
                    for d in detections:
                        obj_name=d["name"]
                        obj_dir=d["direction"]
                        obj_dist=round(d["dist"],1)
                        prev=self.announced_objects.get(obj_name)
                        if prev is None or abs(prev[0]-obj_dist)>0.3 or prev[1]!=obj_dir:
                            speak(f"{obj_name} is {obj_dist} meters {obj_dir}", gtts_codes.get(self.current_language,"en"))
                            self.announced_objects[obj_name]=(obj_dist,obj_dir)
                    self.voice_timer=now
            time.sleep(0.01)

    def get_frame_jpeg(self):
        if self.frame is None: return None
        ret,jpg=cv2.imencode('.jpg',self.frame)
        return jpg.tobytes() if ret else None

detector=Detector()
detector.start()

# ---------------- FLASK APP ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def gen_video():
    while True:
        jpg=detector.get_frame_jpeg()
        if jpg:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+jpg+b'\r\n')
        else:
            time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detections")
def detections():
    return jsonify(detector.last_detections)

@app.route("/toggle_detection", methods=['POST'])
def toggle_detection():
    detector.detection_enabled = not detector.detection_enabled
    return jsonify({"detection_enabled": detector.detection_enabled})

@app.route("/set_language", methods=['POST'])
def set_language():
    data=request.get_json() or {}
    lang=data.get("lang")
    if lang: detector.current_language=lang
    return jsonify({"language": detector.current_language})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
