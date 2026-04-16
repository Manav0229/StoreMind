import os, cv2, math, glob, shutil, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Must be first Streamlit command ──
st.set_page_config(page_title="StoreMind Analytics", page_icon="🏪", layout="wide", initial_sidebar_state="expanded")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
.stApp { background: linear-gradient(170deg, #0a0e1a 0%, #0f1729 40%, #111827 100%); }
h1, h2, h3, h4 { font-family: 'JetBrains Mono', monospace !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #070b14 0%, #0f1729 100%) !important; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
div[data-testid="stMetric"] { background: linear-gradient(145deg, #111827, #1a2234); border: 1px solid #1e293b; border-radius: 14px; padding: 18px 22px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 1.5px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.9rem !important; font-weight: 800; }
button[data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem; }
hr { border-color: #1e293b !important; }
div.stSpinner > div { border-top-color: #06d6a0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS (from your Colab notebook)
# ══════════════════════════════════════════════════════════════

from ultralytics import YOLO

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")


def run_detection(video_path, output_dir, model, conf_threshold=0.25):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    detection_csv = os.path.join(output_dir, f"{video_name}_detections.csv")
    sample_frame_path = os.path.join(output_dir, f"{video_name}_sample_frame.jpg")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    all_detections = []
    frame_idx = 0
    progress = st.progress(0, text="Detecting people...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id == 0 and conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                all_detections.append({"frame": frame_idx, "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "confidence": conf})
        frame_idx += 1
        if frame_count > 0:
            progress.progress(min(frame_idx / frame_count, 1.0), text=f"Detection: {frame_idx}/{frame_count} frames")

    cap.release()
    progress.empty()

    # Save sample frame
    sample_results = model(first_frame, verbose=False)
    sample_annotated = first_frame.copy()
    for box in sample_results[0].boxes:
        if int(box.cls[0].item()) == 0 and float(box.conf[0].item()) >= conf_threshold:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(sample_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(sample_frame_path, sample_annotated)

    df = pd.DataFrame(all_detections)
    df.to_csv(detection_csv, index=False)
    return df


def run_tracking(video_path, output_dir, model, conf=0.25):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    tracks_csv = os.path.join(output_dir, f"{video_name}_tracks.csv")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    track_results = model.track(source=video_path, stream=True, persist=True,
                                 tracker="bytetrack.yaml", classes=[0], conf=conf, verbose=False)
    rows = []
    frame_idx = 0
    progress = st.progress(0, text="Tracking shoppers...")

    for r in track_results:
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else np.array([])
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.array([])
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.full(len(xyxy), -1)
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = box.astype(int)
                cls_id = int(cls[i]) if len(cls) > i else -1
                track_id = int(ids[i]) if len(ids) > i else -1
                score = float(confs[i]) if len(confs) > i else 0.0
                if cls_id == 0:
                    rows.append({"frame": frame_idx, "track_id": track_id, "x1": int(x1), "y1": int(y1),
                                 "x2": int(x2), "y2": int(y2), "width": int(x2-x1), "height": int(y2-y1),
                                 "confidence": score, "cx": int((x1+x2)/2), "cy_bottom": int(y2)})
        frame_idx += 1
        if frame_count > 0:
            progress.progress(min(frame_idx / frame_count, 1.0), text=f"Tracking: {frame_idx}/{frame_count} frames")

    progress.empty()
    df = pd.DataFrame(rows)
    df.to_csv(tracks_csv, index=False)
    return df


def run_counting(video_path, tracks_df, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    candidate_lines = [int(width * r) for r in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]]

    def evaluate_line(line_x):
        df = tracks_df.sort_values(["track_id", "frame"]).reset_index(drop=True)
        counted_ids = set()
        entries, exits = 0, 0
        frame_events = {}
        for tid, group in df.groupby("track_id"):
            group = group.sort_values("frame")
            xs = group["cx"].values
            frames = group["frame"].values
            if len(xs) < 2: continue
            for i in range(1, len(xs)):
                if tid in counted_ids: break
                if xs[i-1] < line_x and xs[i] >= line_x:
                    entries += 1; counted_ids.add(tid); break
                elif xs[i-1] > line_x and xs[i] <= line_x:
                    exits += 1; counted_ids.add(tid); break

        occ = []
        re, rx = 0, 0
        for fr in range(frame_count):
            if fr in frame_events:
                for _, et in frame_events[fr]:
                    if et == "ENTRY": re += 1
                    else: rx += 1
            occ.append(re - rx)
        occ = np.array(occ) if occ else np.array([0])
        penalty = abs(int(occ[-1])) * 5 + abs(min(0, int(np.min(occ)))) * 20
        if int(occ[-1]) < 0: penalty += 1000
        return {"line_x": line_x, "entries": entries, "exits": exits, "score": penalty}

    evals = [evaluate_line(x) for x in candidate_lines]
    eval_df = pd.DataFrame(evals).sort_values("score").reset_index(drop=True)
    best_x = int(eval_df.iloc[0]["line_x"])

    # Rebuild with best line
    df = tracks_df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    counted_ids = set()
    entries, exits = 0, 0
    frame_events = {}
    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        xs = group["cx"].values
        frames = group["frame"].values
        if len(xs) < 2: continue
        for i in range(1, len(xs)):
            if tid in counted_ids: break
            curr_frame = int(frames[i])
            if xs[i-1] < best_x and xs[i] >= best_x:
                entries += 1; counted_ids.add(tid)
                frame_events[curr_frame] = frame_events.get(curr_frame, []) + [(tid, "ENTRY")]; break
            elif xs[i-1] > best_x and xs[i] <= best_x:
                exits += 1; counted_ids.add(tid)
                frame_events[curr_frame] = frame_events.get(curr_frame, []) + [(tid, "EXIT")]; break

    occ_rows = []
    re, rx = 0, 0
    for fr in range(frame_count):
        if fr in frame_events:
            for _, et in frame_events[fr]:
                if et == "ENTRY": re += 1
                else: rx += 1
        occ_rows.append({"frame": fr, "entries_so_far": re, "exits_so_far": rx, "occupancy": re - rx})

    occ_df = pd.DataFrame(occ_rows)
    occ_df.to_csv(os.path.join(output_dir, f"{video_name}_occupancy.csv"), index=False)
    eval_df.to_csv(os.path.join(output_dir, f"{video_name}_line_search.csv"), index=False)
    return occ_df, eval_df, best_x


def run_zone_analytics(video_path, tracks_df, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, first_frame = cap.read()
    cap.release()

    zones = {
        "Left_Aisle": np.array([[0,0],[int(0.33*width),0],[int(0.33*width),height],[0,height]], dtype=np.int32),
        "Center_Aisle": np.array([[int(0.33*width),0],[int(0.66*width),0],[int(0.66*width),height],[int(0.33*width),height]], dtype=np.int32),
        "Right_Aisle": np.array([[int(0.66*width),0],[width,0],[width,height],[int(0.66*width),height]], dtype=np.int32),
        "Checkout_Front": np.array([[0,int(0.78*height)],[width,int(0.78*height)],[width,height],[0,height]], dtype=np.int32),
    }

    def assign_zone(cx, cy):
        if cv2.pointPolygonTest(zones["Checkout_Front"], (float(cx),float(cy)), False) >= 0: return "Checkout_Front"
        for z in ["Left_Aisle","Center_Aisle","Right_Aisle"]:
            if cv2.pointPolygonTest(zones[z], (float(cx),float(cy)), False) >= 0: return z
        return "Unknown"

    tracks_df["zone"] = tracks_df.apply(lambda r: assign_zone(r["cx"], r["cy_bottom"]), axis=1)

    # Dwell
    dwell_rows = []
    for (tid, zone), g in tracks_df.groupby(["track_id","zone"]):
        if zone == "Unknown": continue
        dwell_rows.append({"track_id": int(tid), "zone": zone, "frames_in_zone": len(g), "dwell_time_sec": len(g)/fps if fps > 0 else 0})
    dwell_df = pd.DataFrame(dwell_rows)
    if not dwell_df.empty:
        dwell_summary = dwell_df.groupby("zone").agg(unique_visitors=("track_id","nunique"), total_dwell_sec=("dwell_time_sec","sum"), avg_dwell_sec=("dwell_time_sec","mean")).reset_index().sort_values("total_dwell_sec", ascending=False)
    else:
        dwell_summary = pd.DataFrame(columns=["zone","unique_visitors","total_dwell_sec","avg_dwell_sec"])
    dwell_summary.to_csv(os.path.join(output_dir, f"{video_name}_dwell.csv"), index=False)

    # Transitions
    transitions = []
    for tid, g in tracks_df.groupby("track_id"):
        seq = [z for z in g.sort_values("frame")["zone"].tolist() if z != "Unknown"]
        compressed = [seq[0]] if seq else []
        for z in seq[1:]:
            if z != compressed[-1]: compressed.append(z)
        for i in range(len(compressed)-1): transitions.append((compressed[i], compressed[i+1]))
    zone_names = list(zones.keys())
    if transitions:
        tdf = pd.DataFrame(transitions, columns=["from","to"])
        trans_matrix = pd.crosstab(tdf["from"], tdf["to"])
    else:
        trans_matrix = pd.DataFrame(0, index=zone_names, columns=zone_names)
    for z in zone_names:
        if z not in trans_matrix.index: trans_matrix.loc[z] = 0
        if z not in trans_matrix.columns: trans_matrix[z] = 0
    trans_matrix = trans_matrix.reindex(index=zone_names, columns=zone_names, fill_value=0)
    trans_matrix.to_csv(os.path.join(output_dir, f"{video_name}_transitions.csv"))

    # Heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    for _, r in tracks_df.iterrows():
        x = int(np.clip(r["cx"], 0, width-1))
        y = int(np.clip(r["cy_bottom"], 0, height-1))
        heatmap[y, x] += 1.0
    heatmap_blur = cv2.GaussianBlur(heatmap, (0,0), sigmaX=25, sigmaY=25)
    if heatmap_blur.max() > 0:
        heatmap_norm = (heatmap_blur / heatmap_blur.max() * 255).astype(np.uint8)
    else:
        heatmap_norm = heatmap_blur.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(first_frame, 0.55, heatmap_color, 0.45, 0)
    heatmap_path = os.path.join(output_dir, f"{video_name}_heatmap.png")
    cv2.imwrite(heatmap_path, overlay)

    # Zone overlay
    zone_img = first_frame.copy()
    zcolors = {"Left_Aisle":(255,0,0),"Center_Aisle":(0,255,0),"Right_Aisle":(0,0,255),"Checkout_Front":(255,255,0)}
    for zn, poly in zones.items():
        cv2.polylines(zone_img, [poly], True, zcolors[zn], 3)
        cx_t, cy_t = int(np.mean(poly[:,0])), int(np.mean(poly[:,1]))
        cv2.putText(zone_img, zn, (cx_t-60, cy_t), cv2.FONT_HERSHEY_SIMPLEX, 0.8, zcolors[zn], 2)
    zone_path = os.path.join(output_dir, f"{video_name}_zones.png")
    cv2.imwrite(zone_path, zone_img)

    # Trajectory map
    traj_img = first_frame.copy()
    rng_colors = np.random.randint(0, 255, (1000, 3))
    for tid, g in tracks_df.groupby("track_id"):
        pts = list(zip(g["cx"], g["cy_bottom"]))
        color = tuple(int(x) for x in rng_colors[int(tid)%1000])
        for i in range(1, len(pts)):
            cv2.line(traj_img, (int(pts[i-1][0]),int(pts[i-1][1])), (int(pts[i][0]),int(pts[i][1])), color, 2)
    traj_path = os.path.join(output_dir, f"{video_name}_trajectories.png")
    cv2.imwrite(traj_path, traj_img)

    return dwell_summary, trans_matrix, heatmap_path, zone_path, traj_path


def run_congestion(dwell_summary):
    if dwell_summary.empty: return pd.DataFrame()
    avg = dwell_summary["avg_dwell_sec"].mean()
    rows = []
    for _, r in dwell_summary.iterrows():
        score = r["avg_dwell_sec"] / avg if avg > 0 else 0
        level = "High congestion risk" if score > 1.4 else "Moderate congestion" if score > 1.0 else "Low congestion"
        rows.append({"zone": r["zone"], "avg_dwell_sec": r["avg_dwell_sec"], "congestion_score": round(score, 2), "congestion_level": level})
    return pd.DataFrame(rows)


def run_layout_optimizer(dwell_summary, congestion_df, trans_matrix):
    if dwell_summary.empty: return pd.DataFrame()
    df = pd.merge(dwell_summary, congestion_df[["zone","congestion_level"]], on="zone", how="left")
    results = []
    for _, r in df.iterrows():
        recs = []
        if r["total_dwell_sec"] == df["total_dwell_sec"].max():
            recs += ["Place promotions in high traffic zone", "Feature premium products here"]
        if r["total_dwell_sec"] == df["total_dwell_sec"].min():
            recs += ["Move shelf closer to entrance", "Improve signage or product visibility"]
        cl = str(r.get("congestion_level",""))
        if "High" in cl: recs += ["Reduce shelf density", "Increase walking space"]
        elif "Moderate" in cl: recs.append("Monitor congestion and adjust layout")
        if r["zone"] in trans_matrix.index:
            flows = trans_matrix.loc[r["zone"]].drop(r["zone"], errors="ignore")
            if len(flows) > 0 and flows.max() > 10:
                recs.append(f"Cross-sell between {r['zone']} and {flows.idxmax()}")
        if not recs: recs.append("Layout performing normally")
        results.append({"zone": r["zone"], "total_dwell_sec": round(r["total_dwell_sec"],2), "congestion_level": cl, "recommendations": " | ".join(recs)})
    return pd.DataFrame(results)


def run_energy(occ_df, dwell_summary, congestion_df):
    if occ_df.empty: return pd.DataFrame(), pd.DataFrame()
    avg_occ = float(occ_df["occupancy"].mean())
    peak_occ = max(int(occ_df["occupancy"].max()), 1)
    ratio = avg_occ / peak_occ
    if ratio >= 0.80: hf,lf,temp,fan,light,mode = 1.0,1.0,22.0,100,100,"High Occupancy"
    elif ratio >= 0.55: hf,lf,temp,fan,light,mode = 0.92,0.95,22.5,90,95,"Normal"
    elif ratio >= 0.30: hf,lf,temp,fan,light,mode = 0.82,0.85,23.5,75,85,"Energy Saving"
    else: hf,lf,temp,fan,light,mode = 0.70,0.75,24.0,65,75,"Deep Energy Saving"
    bh, bl = 20.0, 8.0
    energy_df = pd.DataFrame([{"system_mode":mode,"baseline_hvac_kw":bh,"baseline_lighting_kw":bl,"baseline_total_kw":bh+bl,
        "optimized_hvac_kw":round(bh*hf,2),"optimized_lighting_kw":round(bl*lf,2),"optimized_total_kw":round(bh*hf+bl*lf,2),
        "estimated_total_savings_pct":round((1-(bh*hf+bl*lf)/(bh+bl))*100,2),
        "ahu_supply_temp_setpoint_c":temp,"ahu_fan_speed_pct":fan,"store_lighting_level_pct":light,
        "avg_occupancy":round(avg_occ,2),"peak_occupancy":peak_occ}])

    # Zone controls
    if not dwell_summary.empty and not congestion_df.empty:
        zdf = dwell_summary.merge(congestion_df[["zone","congestion_level"]], on="zone", how="left")
        cmds = []
        for _, r in zdf.iterrows():
            zl,za,zt,cm = 85,80,23.0,"Normal"
            if r["total_dwell_sec"] == zdf["total_dwell_sec"].max(): zl,za,zt,cm = 100,100,22.0,"High Traffic Priority"
            elif r["total_dwell_sec"] == zdf["total_dwell_sec"].min(): zl,za,zt,cm = 70,65,24.0,"Low Traffic Energy Save"
            cg = str(r.get("congestion_level",""))
            if "High" in cg: za,zt,cm = max(za,100),min(zt,21.5),"Congestion Relief"
            elif "Moderate" in cg: za,zt = max(za,90),min(zt,22.0)
            cmds.append({"zone":r["zone"],"recommended_lighting_pct":zl,"recommended_airflow_pct":za,"recommended_temp_setpoint_c":zt,"control_mode":cm})
        controls_df = pd.DataFrame(cmds)
    else:
        controls_df = pd.DataFrame()
    return energy_df, controls_df


# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════
def dark(fig, h=400):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
        font=dict(family="Outfit", color="#cbd5e1"), title_font=dict(family="JetBrains Mono", size=16, color="#f1f5f9"),
        height=h, margin=dict(l=40,r=20,t=50,b=40), xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"))
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:10px 0 20px"><span style="font-size:2.5rem">🏪</span><h2 style="margin:4px 0 0;font-size:1.4rem;color:#06d6a0">StoreMind</h2><p style="margin:2px 0;font-size:0.7rem;color:#64748b;letter-spacing:2px;text-transform:uppercase">Retail Intelligence</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📹 Upload Video")
    uploaded = st.file_uploader("Upload a store/retail video", type=["mp4","avi","mov","mkv"])

    use_sample = st.checkbox("Or use sample video", value=False,
                              help="Uses the pre-downloaded YouTube video if available at /content/storemind_input_video.mp4")

    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)

    st.markdown("---")
    run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**Pipeline**\n\n1. 🔍 YOLOv8 Detection\n2. 🏃 ByteTrack Tracking\n3. 🚪 Entry/Exit Counting\n4. 📍 Zone Analytics\n5. 🔥 Heatmap\n6. ⚠️ Congestion\n7. 🗺️ Layout AI\n8. ⚡ Energy BMS")
    st.markdown("---")
    st.caption("EECE 7370 — Advanced Computer Vision")
    st.caption("Manav Sanjay Singh · Spring 2025")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

st.markdown('<h1 style="font-size:2rem;color:#f1f5f9">🏪 StoreMind <span style="color:#06d6a0">Analytics Dashboard</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b;font-size:0.88rem">Upload any retail video → get full crowd analytics & optimization insights powered by YOLOv8</p>', unsafe_allow_html=True)
st.markdown("")

# Session state for results
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if run_btn:
    # Determine video path
    video_path = None
    if uploaded is not None:
        tmp = os.path.join("/content", uploaded.name)
        with open(tmp, "wb") as vf:
            vf.write(uploaded.getbuffer())
        video_path = tmp
    elif use_sample and os.path.exists("/content/storemind_input_video.mp4"):
        video_path = "/content/storemind_input_video.mp4"
    else:
        st.error("Please upload a video or check 'Use sample video'")
        st.stop()

    output_dir = "/content/storemind_outputs"
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    model = load_yolo()

    # Run pipeline
    st.markdown("### ⏳ Running StoreMind Pipeline...")

    with st.status("🔍 Step 1/7: YOLOv8 Detection...", expanded=True) as status:
        det_df = run_detection(video_path, output_dir, model, conf_threshold)
        status.update(label=f"✅ Detection complete — {len(det_df)} detections", state="complete")

    with st.status("🏃 Step 2/7: ByteTrack Tracking...", expanded=True) as status:
        tracks_df = run_tracking(video_path, output_dir, model, conf_threshold)
        status.update(label=f"✅ Tracking complete — {tracks_df['track_id'].nunique() if not tracks_df.empty else 0} shoppers", state="complete")

    with st.status("🚪 Step 3/7: Entry/Exit Counting...", expanded=True) as status:
        occ_df, line_df, best_line = run_counting(video_path, tracks_df, output_dir)
        status.update(label=f"✅ Counting complete — best line x={best_line}", state="complete")

    with st.status("📍 Step 4/7: Zone Analytics...", expanded=True) as status:
        dwell, trans, heatmap_path, zone_path, traj_path = run_zone_analytics(video_path, tracks_df, output_dir)
        status.update(label=f"✅ Zone analytics complete — {len(dwell)} zones", state="complete")

    with st.status("⚠️ Step 5/7: Congestion Analysis...", expanded=True) as status:
        congestion = run_congestion(dwell)
        status.update(label="✅ Congestion analysis complete", state="complete")

    with st.status("🗺️ Step 6/7: Layout Optimization...", expanded=True) as status:
        layout = run_layout_optimizer(dwell, congestion, trans)
        status.update(label="✅ Layout optimization complete", state="complete")

    with st.status("⚡ Step 7/7: Energy BMS...", expanded=True) as status:
        energy, controls = run_energy(occ_df, dwell, congestion)
        status.update(label="✅ Energy optimization complete", state="complete")

    # Store results
    st.session_state.results_ready = True
    st.session_state.tracks_df = tracks_df
    st.session_state.occ_df = occ_df
    st.session_state.line_df = line_df
    st.session_state.dwell = dwell
    st.session_state.trans = trans
    st.session_state.congestion = congestion
    st.session_state.layout = layout
    st.session_state.energy = energy
    st.session_state.controls = controls
    st.session_state.heatmap_path = heatmap_path
    st.session_state.zone_path = zone_path
    st.session_state.traj_path = traj_path

    st.success("🎉 Pipeline complete! Scroll down for results.")
    st.balloons()


# ══════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════

if st.session_state.results_ready:
    S = st.session_state
    tracks, occ, dwell = S.tracks_df, S.occ_df, S.dwell

    # KPIs
    us = int(tracks["track_id"].nunique()) if not tracks.empty else 0
    ap = float(tracks.groupby("frame").size().mean()) if not tracks.empty else 0
    te = int(occ["entries_so_far"].max()) if not occ.empty else 0
    tx = int(occ["exits_so_far"].max()) if not occ.empty else 0
    po = int(occ["occupancy"].max()) if not occ.empty else 0
    ao = float(occ["occupancy"].mean()) if not occ.empty else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Unique Shoppers", us); c2.metric("Avg/Frame", f"{ap:.1f}"); c3.metric("Peak Occ.", po)
    c4.metric("Entries", te); c5.metric("Exits", tx); c6.metric("Avg Occ.", f"{ao:.1f}")
    st.markdown("")

    # Tabs
    tab_occ, tab_zones, tab_heat, tab_cong, tab_layout, tab_energy = st.tabs(["📊 Occupancy","📍 Zones","🔥 Heatmap","⚠️ Congestion","🗺️ Layout AI","⚡ Energy"])

    with tab_occ:
        st.markdown("### Occupancy Over Time")
        if not occ.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=occ["frame"],y=occ["occupancy"],mode="lines",line=dict(color="#06d6a0",width=2.5),fill="tozeroy",fillcolor="rgba(6,214,160,0.07)",name="Occupancy"))
            fig.add_trace(go.Scatter(x=occ["frame"],y=occ["entries_so_far"],mode="lines",line=dict(color="#3b82f6",width=1.5,dash="dot"),name="Entries"))
            fig.add_trace(go.Scatter(x=occ["frame"],y=occ["exits_so_far"],mode="lines",line=dict(color="#f59e0b",width=1.5,dash="dot"),name="Exits"))
            dark(fig,420); fig.update_layout(xaxis_title="Frame",yaxis_title="Count",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig, use_container_width=True)
        if not S.line_df.empty:
            with st.expander("🔎 Line Search Results"):
                st.dataframe(S.line_df, use_container_width=True)

    with tab_zones:
        st.markdown("### Zone Dwell Time")
        if not dwell.empty:
            colors = ["#06d6a0","#3b82f6","#8b5cf6","#f59e0b","#ef4444","#ec4899"]
            ca,cb = st.columns(2)
            with ca:
                fig_d = go.Figure()
                for i,(_,r) in enumerate(dwell.iterrows()):
                    fig_d.add_trace(go.Bar(x=[r["zone"]],y=[r["total_dwell_sec"]],marker_color=colors[i%len(colors)],text=f'{r["total_dwell_sec"]:.1f}s',textposition="outside"))
                dark(fig_d,380); fig_d.update_layout(title="Total Dwell by Zone",showlegend=False,bargap=0.3); st.plotly_chart(fig_d, use_container_width=True)
            with cb:
                if "unique_visitors" in dwell.columns:
                    fig_v = go.Figure()
                    for i,(_,r) in enumerate(dwell.iterrows()):
                        fig_v.add_trace(go.Bar(x=[r["zone"]],y=[r["unique_visitors"]],marker_color=colors[i%len(colors)],text=str(int(r["unique_visitors"])),textposition="outside"))
                    dark(fig_v,380); fig_v.update_layout(title="Visitors by Zone",showlegend=False,bargap=0.3); st.plotly_chart(fig_v, use_container_width=True)
            st.dataframe(dwell, use_container_width=True)
        st.markdown("### Transition Matrix")
        trans = S.trans
        if not trans.empty:
            fig_t = go.Figure(data=go.Heatmap(z=trans.values,x=trans.columns.tolist(),y=trans.index.tolist(),colorscale=[[0,"#111827"],[0.5,"#3b82f6"],[1,"#06d6a0"]],text=trans.values.astype(int),texttemplate="%{text}",textfont=dict(size=14,color="white")))
            dark(fig_t,400); fig_t.update_layout(title="Customer Flow"); st.plotly_chart(fig_t, use_container_width=True)

    with tab_heat:
        ch,ct = st.columns(2)
        with ch:
            st.markdown("### Density Heatmap")
            if os.path.exists(S.heatmap_path): st.image(S.heatmap_path, use_container_width=True)
        with ct:
            st.markdown("### Trajectories")
            if os.path.exists(S.traj_path): st.image(S.traj_path, use_container_width=True)
        if os.path.exists(S.zone_path):
            st.markdown("### Zone Overlay")
            st.image(S.zone_path, use_container_width=True)

    with tab_cong:
        st.markdown("### Congestion Analysis")
        cong = S.congestion
        if not cong.empty:
            cols = st.columns(len(cong))
            for i,(_,r) in enumerate(cong.iterrows()):
                lev = str(r.get("congestion_level",""))
                sc = r.get("congestion_score",0)
                z = r.get("zone","")
                if "High" in lev: col,ico = "#ef4444","🔴"
                elif "Moderate" in lev: col,ico = "#f59e0b","🟡"
                else: col,ico = "#06d6a0","🟢"
                with cols[i]:
                    st.markdown(f'<div style="background:linear-gradient(145deg,#111827,#1a2234);border:1px solid {col}33;border-radius:14px;padding:20px;text-align:center"><div style="font-size:2rem">{ico}</div><div style="font-family:JetBrains Mono,monospace;color:#f1f5f9;margin:8px 0 4px">{z}</div><div style="color:{col};font-size:0.85rem;font-weight:600">{lev}</div><div style="color:#64748b;font-size:0.75rem;margin-top:4px">Score: {sc}</div></div>', unsafe_allow_html=True)

    with tab_layout:
        st.markdown("### Layout Optimization")
        layout = S.layout
        if not layout.empty:
            for _,r in layout.iterrows():
                z,dv,cl,rt = r.get("zone",""),r.get("total_dwell_sec",0),str(r.get("congestion_level","")),str(r.get("recommendations",""))
                bd = "#ef4444" if "High" in cl else "#f59e0b" if "Moderate" in cl else "#06d6a0"
                rh = "".join(f'<div style="color:#94a3b8;font-size:0.86rem;margin:3px 0;padding-left:12px;border-left:2px solid #1e293b">{x.strip()}</div>' for x in rt.split("|") if x.strip())
                st.markdown(f'<div style="background:linear-gradient(145deg,#111827,#1a2234);border-left:4px solid {bd};border-radius:0 12px 12px 0;padding:18px 22px;margin-bottom:12px"><div style="display:flex;justify-content:space-between"><span style="font-family:JetBrains Mono,monospace;font-size:1.05rem;color:#f1f5f9">{z}</span><span style="color:#64748b;font-size:0.8rem">Dwell: {dv:.1f}s | {cl}</span></div><div style="margin-top:10px">{rh}</div></div>', unsafe_allow_html=True)

    with tab_energy:
        st.markdown("### BMS Energy Optimization")
        energy = S.energy; controls = S.controls
        if not energy.empty:
            e = energy.iloc[0]
            e1,e2,e3,e4 = st.columns(4)
            e1.metric("Mode",str(e.get("system_mode","?"))); e2.metric("Baseline",f'{e.get("baseline_total_kw",0):.1f} kW')
            e3.metric("Optimized",f'{e.get("optimized_total_kw",0):.1f} kW'); e4.metric("Savings",f'{e.get("estimated_total_savings_pct",0):.1f}%')
            st.markdown("")
            cats=["HVAC","Lighting","Total"]
            base=[e.get("baseline_hvac_kw",0),e.get("baseline_lighting_kw",0),e.get("baseline_total_kw",0)]
            opt=[e.get("optimized_hvac_kw",0),e.get("optimized_lighting_kw",0),e.get("optimized_total_kw",0)]
            fig_e=go.Figure()
            fig_e.add_trace(go.Bar(x=cats,y=base,name="Baseline",marker_color="#ef4444",text=[f"{v:.1f}" for v in base],textposition="outside"))
            fig_e.add_trace(go.Bar(x=cats,y=opt,name="Optimized",marker_color="#06d6a0",text=[f"{v:.1f}" for v in opt],textposition="outside"))
            dark(fig_e,380); fig_e.update_layout(title="Baseline vs Optimized (kW)",barmode="group"); st.plotly_chart(fig_e, use_container_width=True)
            if not controls.empty:
                st.markdown("### Zone BMS Controls")
                for _,r in controls.iterrows():
                    z,m = r.get("zone",""),str(r.get("control_mode",""))
                    l,a,t = r.get("recommended_lighting_pct",0),r.get("recommended_airflow_pct",0),r.get("recommended_temp_setpoint_c",0)
                    if "Congestion" in m: ico,col = "🌀","#ef4444"
                    elif "High" in m: ico,col = "🔥","#f59e0b"
                    elif "Low" in m: ico,col = "💤","#06d6a0"
                    else: ico,col = "⚙️","#3b82f6"
                    st.markdown(f'<div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:14px 18px;margin-bottom:8px;display:flex;align-items:center;gap:16px"><span style="font-size:1.5rem">{ico}</span><div style="flex:1"><div style="font-family:JetBrains Mono,monospace;color:#f1f5f9">{z}</div><div style="color:{col};font-size:0.78rem;font-weight:600">{m}</div></div><div style="display:flex;gap:20px;color:#94a3b8;font-size:0.82rem"><span>💡 {l}%</span><span>🌬️ {a}%</span><span>🌡️ {t}°C</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="text-align:center;padding:10px 0;color:#475569;font-size:0.78rem"><strong style="color:#06d6a0">StoreMind</strong> — EECE 7370 Advanced Computer Vision | YOLOv8 + ByteTrack + Streamlit</div>', unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("")
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; background:linear-gradient(145deg,#111827,#1a2234); border-radius:16px; border:1px solid #1e293b; margin:20px 0;">
        <div style="font-size:4rem; margin-bottom:16px;">📹</div>
        <h2 style="color:#f1f5f9; margin:0;">Upload a Store Video to Begin</h2>
        <p style="color:#64748b; margin:12px 0 0; font-size:0.95rem;">
            Use the sidebar to upload any retail/store video (MP4, AVI, MOV)<br>
            or check "Use sample video" if you've already downloaded one.<br><br>
            Then click <strong style="color:#06d6a0;">Run Full Analysis</strong> to process it.
        </p>
    </div>
    """, unsafe_allow_html=True)
