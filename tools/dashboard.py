# -*- coding: utf-8 -*-
"""dashboard.py —— 轻量级零依赖本地训练 & GPU 实时监控网页

用法：
    python tools/dashboard.py --port 8765 --run m12l_d384_full1125_ep2
浏览器打开：
    http://127.0.0.1:8765
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_ROOT = os.path.join(ROOT, "out")

GPU_HIST = collections.deque(maxlen=300)  # 最近 ~10 分钟（每 2s 采样一次）
GPU_LOCK = threading.Lock()
DEFAULT_RUN = "m12l_d384_full1125_ep2"


def query_gpu():
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,power.draw,power.limit,fan.speed,clocks.gr",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=3,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        line = out.decode("utf-8", errors="replace").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        def fnum(s):
            try:
                return float(s)
            except Exception:
                return None
        return {
            "ok": True,
            "t": time.strftime("%H:%M:%S"),
            "ts": time.time(),
            "name": parts[0],
            "temp": fnum(parts[1]),
            "util_gpu": fnum(parts[2]),
            "util_mem": fnum(parts[3]),
            "mem_used": fnum(parts[4]),
            "mem_total": fnum(parts[5]),
            "power_draw": fnum(parts[6]),
            "power_limit": fnum(parts[7]),
            "fan": fnum(parts[8]),
            "clock": fnum(parts[9]),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "t": time.strftime("%H:%M:%S"), "ts": time.time()}


def gpu_poller():
    while True:
        g = query_gpu()
        with GPU_LOCK:
            GPU_HIST.append(g)
        time.sleep(2.0)


def list_runs():
    if not os.path.isdir(OUT_ROOT):
        return []
    runs = []
    for name in os.listdir(OUT_ROOT):
        p = os.path.join(OUT_ROOT, name)
        if os.path.isdir(p) and (os.path.isfile(os.path.join(p, "metrics.jsonl"))
                                 or os.path.isfile(os.path.join(p, "log.txt"))):
            mt = os.path.getmtime(p)
            st_p = os.path.join(p, "status.json")
            if os.path.isfile(st_p):
                mt = max(mt, os.path.getmtime(st_p))
            runs.append((mt, name))
    runs.sort(reverse=True)
    return [n for _, n in runs]


def read_run_state(run_name):
    rdir = os.path.join(OUT_ROOT, run_name)
    if not os.path.isdir(rdir):
        return {"ok": False, "error": f"run not found: {run_name}"}

    status = {}
    st_path = os.path.join(rdir, "status.json")
    if os.path.isfile(st_path):
        try:
            with open(st_path, encoding="utf-8") as f:
                status = json.load(f)
        except Exception:
            pass

    meta = {}
    meta_path = os.path.join(rdir, "run_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    train_pts = []
    val_pts = []
    m_path = os.path.join(rdir, "metrics.jsonl")
    if os.path.isfile(m_path):
        try:
            with open(m_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if "val_loss" in rec:
                        val_pts.append({
                            "step": rec.get("step", 0),
                            "val_loss": rec.get("val_loss"),
                            "moe_summary": rec.get("moe_summary", ""),
                            "t": rec.get("t", ""),
                        })
                    elif "loss" in rec:
                        train_pts.append({
                            "step": rec.get("step", 0),
                            "epoch": rec.get("epoch", 0),
                            "loss": rec.get("loss"),
                            "lr": rec.get("lr"),
                            "grad_norm": rec.get("grad_norm"),
                            "tok_s": rec.get("tok_s"),
                            "gpu_mem_mb": rec.get("gpu_mem_mb"),
                            "gpu_peak_mb": rec.get("gpu_peak_mb"),
                            "t": rec.get("t", ""),
                        })
        except Exception:
            pass

    # 若 train_pts 太多，做轻度下采样保证前端渲染丝滑（保留首尾）
    if len(train_pts) > 800:
        stride = max(1, len(train_pts) // 600)
        sampled = train_pts[::stride]
        if sampled[-1] is not train_pts[-1]:
            sampled.append(train_pts[-1])
        train_pts_out = sampled
    else:
        train_pts_out = train_pts

    # 尾部日志
    log_tail = []
    log_path = os.path.join(rdir, "log.txt")
    if os.path.isfile(log_path):
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            log_tail = [l.rstrip("\r\n") for l in lines[-35:]]
        except Exception:
            pass

    # 最新采样
    samples = []
    sdir = os.path.join(rdir, "samples")
    if os.path.isdir(sdir):
        sfiles = sorted([f for f in os.listdir(sdir) if f.endswith(".txt")], reverse=True)
        for sf in sfiles[:6]:
            try:
                with open(os.path.join(sdir, sf), encoding="utf-8", errors="replace") as f:
                    samples.append({"name": sf, "text": f.read()})
            except Exception:
                pass

    with GPU_LOCK:
        gpu_hist = list(GPU_HIST)
        gpu_now = gpu_hist[-1] if gpu_hist else query_gpu()

    return {
        "ok": True,
        "run": run_name,
        "runs": list_runs(),
        "status": status,
        "meta": meta,
        "train": train_pts_out,
        "train_total_points": len(train_pts),
        "val": val_pts,
        "gpu": gpu_now,
        "gpu_hist": gpu_hist[-120:],
        "log_tail": log_tail,
        "samples": samples,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练 & GPU 实时监控台</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #262c36;
    --text: #e6edf3; --muted: #8b949e;
    --blue: #58a6ff; --green: #3fb950; --orange: #f0883e;
    --red: #f85149; --purple: #bc8cff; --cyan: #39c5cf;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    padding: 16px 20px 32px; line-height: 1.45;
  }
  .topbar {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 12px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
  }
  .title { font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); }
  .controls { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); }
  select, button {
    background: var(--card); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 10px; font-size: 13px; cursor: pointer;
  }
  select:hover, button:hover { border-color: var(--blue); }
  .progress-wrap {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; margin-bottom: 14px;
  }
  .progress-head { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; font-size: 14px; }
  .bar-bg { position: relative; height: 12px; background: #21262d; border-radius: 6px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #1f6feb, #3fb950); transition: width 0.4s; }
  .bar-ep-mark { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: rgba(255,255,255,0.45); }
  .grid-kpi {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 10px; margin-bottom: 14px;
  }
  .kpi {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px;
  }
  .kpi-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .kpi-val { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .kpi-sub { font-size: 11px; color: var(--muted); margin-top: 3px; }
  .c-blue { color: var(--blue); } .c-green { color: var(--green); }
  .c-orange { color: var(--orange); } .c-red { color: var(--red); }
  .c-purple { color: var(--purple); } .c-cyan { color: var(--cyan); }
  .grid-charts {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 12px; margin-bottom: 14px;
  }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px;
  }
  .card-title {
    font-size: 14px; font-weight: 600; margin-bottom: 10px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .legend { display: flex; gap: 12px; font-size: 12px; color: var(--muted); font-weight: 400; }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .sw { width: 10px; height: 3px; display: inline-block; border-radius: 2px; }
  canvas { width: 100%; height: 220px; display: block; }
  .grid-bottom {
    display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px;
  }
  @media (max-width: 960px) { .grid-bottom { grid-template-columns: 1fr; } }
  pre {
    background: #0b0f14; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; font-size: 12px; line-height: 1.5;
    overflow: auto; max-height: 300px; white-space: pre-wrap; word-break: break-all;
    font-family: Consolas, "Courier New", monospace; color: #c9d1d9;
  }
  .sample-tabs { display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }
  .sample-tab {
    font-size: 12px; padding: 3px 8px; border-radius: 5px;
    background: #21262d; border: 1px solid var(--border); color: var(--muted); cursor: pointer;
  }
  .sample-tab.active { background: #1f6feb33; border-color: var(--blue); color: var(--text); }
</style>
</head>
<body>
<div class="topbar">
  <div class="title"><span class="dot" id="liveDot"></span>训练 &amp; GPU 实时监控台</div>
  <div class="controls">
    <span>任务：</span>
    <select id="runSel"></select>
    <span id="clockTxt">--:--:--</span>
    <button id="pauseBtn">暂停刷新</button>
  </div>
</div>

<div class="progress-wrap">
  <div class="progress-head">
    <div><strong id="runName">-</strong> <span style="color:var(--muted)" id="epochTxt">Epoch -</span></div>
    <div id="stepTxt">Step - / - (-%) ｜ ETA -</div>
  </div>
  <div class="bar-bg">
    <div class="bar-fill" id="progBar" style="width:0%"></div>
    <div class="bar-ep-mark" title="Epoch 1 / Epoch 2 分界"></div>
  </div>
</div>

<div class="grid-kpi">
  <div class="kpi"><div class="kpi-label">Train Loss</div><div class="kpi-val c-blue" id="kLoss">-</div><div class="kpi-sub" id="kGrad">grad -</div></div>
  <div class="kpi"><div class="kpi-label">Best Val Loss</div><div class="kpi-val c-green" id="kBestVal">-</div><div class="kpi-sub" id="kLastVal">最新 val -</div></div>
  <div class="kpi"><div class="kpi-label">吞吐速度</div><div class="kpi-val c-cyan" id="kTps">-</div><div class="kpi-sub" id="kLr">LR -</div></div>
  <div class="kpi"><div class="kpi-label">GPU 温度</div><div class="kpi-val" id="kTemp">-</div><div class="kpi-sub" id="kFan">风扇 - ｜ 频率 -</div></div>
  <div class="kpi"><div class="kpi-label">GPU 功耗 &amp; 利用率</div><div class="kpi-val c-purple" id="kPower">-</div><div class="kpi-sub" id="kUtil">GPU -% ｜ 显存带宽 -%</div></div>
  <div class="kpi"><div class="kpi-label">显存占用</div><div class="kpi-val c-orange" id="kVram">-</div><div class="kpi-sub" id="kPeak">训练峰值 -</div></div>
</div>

<div class="grid-charts">
  <div class="card">
    <div class="card-title">
      <span>Loss 曲线（Train &amp; Val）</span>
      <div class="legend">
        <span><i class="sw" style="background:#58a6ff"></i>Train Loss</span>
        <span><i class="sw" style="background:#3fb950"></i>Val Loss</span>
      </div>
    </div>
    <canvas id="cLoss"></canvas>
  </div>
  <div class="card">
    <div class="card-title">
      <span>GPU 实时温度 &amp; 功耗（最近 4 分钟）</span>
      <div class="legend">
        <span><i class="sw" style="background:#f85149"></i>温度 (°C)</span>
        <span><i class="sw" style="background:#bc8cff"></i>功耗 (W)</span>
      </div>
    </div>
    <canvas id="cGpu"></canvas>
  </div>
  <div class="card">
    <div class="card-title">
      <span>吞吐速度 (tok/s) &amp; 梯度范数</span>
      <div class="legend">
        <span><i class="sw" style="background:#39c5cf"></i>tok/s</span>
        <span><i class="sw" style="background:#f0883e"></i>Grad Norm</span>
      </div>
    </div>
    <canvas id="cSpeed"></canvas>
  </div>
  <div class="card">
    <div class="card-title">
      <span>学习率 (LR) 变化曲线</span>
      <div class="legend">
        <span><i class="sw" style="background:#d2a8ff"></i>LR</span>
      </div>
    </div>
    <canvas id="cLr"></canvas>
  </div>
</div>

<div class="grid-bottom">
  <div class="card">
    <div class="card-title"><span>实时日志（最近 35 行）</span><span style="font-size:12px;color:var(--muted)" id="gpuNameTxt">-</span></div>
    <pre id="logBox">加载中...</pre>
  </div>
  <div class="card">
    <div class="card-title"><span>模型生成采样预览</span></div>
    <div class="sample-tabs" id="sampleTabs"></div>
    <pre id="sampleBox">暂未生成采样（每 2000 步自动输出一次）</pre>
  </div>
</div>

<script>
let currentRun = new URLSearchParams(location.search).get("run") || "";
let paused = false;
let activeSampleIdx = 0;

document.getElementById("pauseBtn").onclick = () => {
  paused = !paused;
  document.getElementById("pauseBtn").textContent = paused ? "恢复刷新" : "暂停刷新";
  document.getElementById("liveDot").style.background = paused ? "#8b949e" : "#3fb950";
};
document.getElementById("runSel").onchange = (e) => {
  currentRun = e.target.value;
  activeSampleIdx = 0;
  fetchData();
};

function setupCanvas(id) {
  const c = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.max(300, Math.floor(rect.width * dpr));
  c.height = Math.max(160, Math.floor(rect.height * dpr));
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w: rect.width, h: rect.height };
}

function drawChart(id, seriesList, opts = {}) {
  const { ctx, w, h } = setupCanvas(id);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 50, r: opts.dual ? 46 : 16, t: 12, b: 26 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

  // 收集 X 范围
  let xs = [];
  seriesList.forEach(s => s.pts.forEach(p => xs.push(p.x)));
  if (xs.length === 0) {
    ctx.fillStyle = "#8b949e"; ctx.font = "13px sans-serif";
    ctx.fillText("等待训练数据写入...", pad.l + 10, pad.t + ph / 2);
    return;
  }
  const xmin = Math.min(...xs), xmax = Math.max(...xs, xmin + 1);

  // 计算左右 Y 轴范围
  function yRange(axis) {
    let ys = [];
    seriesList.filter(s => (s.axis || 0) === axis).forEach(s => s.pts.forEach(p => {
      if (p.y != null && isFinite(p.y)) ys.push(p.y);
    }));
    if (!ys.length) return [0, 1];
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    const m = (hi - lo) * 0.08;
    return [lo - m, hi + m];
  }
  const y0 = yRange(0), y1 = opts.dual ? yRange(1) : y0;

  // 网格
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  ctx.font = "11px Consolas, monospace"; ctx.fillStyle = "#8b949e";
  for (let i = 0; i <= 4; i++) {
    const yy = pad.t + (ph * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
    const v0 = y0[1] - ((y0[1] - y0[0]) * i) / 4;
    ctx.textAlign = "right";
    ctx.fillText(opts.fmt0 ? opts.fmt0(v0) : v0.toFixed(2), pad.l - 6, yy + 4);
    if (opts.dual) {
      const v1 = y1[1] - ((y1[1] - y1[0]) * i) / 4;
      ctx.textAlign = "left";
      ctx.fillText(opts.fmt1 ? opts.fmt1(v1) : v1.toFixed(1), pad.l + pw + 6, yy + 4);
    }
  }
  // X 轴刻度
  ctx.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const xx = pad.l + (pw * i) / 4;
    const vx = xmin + ((xmax - xmin) * i) / 4;
    ctx.fillText(opts.fmtX ? opts.fmtX(vx) : Math.round(vx), xx, h - 6);
  }
  // 可选竖线（例如 Epoch 1->2）
  if (opts.vlineX != null && opts.vlineX > xmin && opts.vlineX < xmax) {
    const vx = pad.l + ((opts.vlineX - xmin) / (xmax - xmin)) * pw;
    ctx.save();
    ctx.setLineDash([4, 4]); ctx.strokeStyle = "#8b949e88";
    ctx.beginPath(); ctx.moveTo(vx, pad.t); ctx.lineTo(vx, pad.t + ph); ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.fillText("Ep2", vx + 14, pad.t + 12);
    ctx.restore();
  }

  // 画曲线
  seriesList.forEach(s => {
    const yr = (s.axis || 0) === 1 ? y1 : y0;
    const pts = s.pts.filter(p => p.y != null && isFinite(p.y));
    if (!pts.length) return;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2;
    ctx.beginPath();
    pts.forEach((p, idx) => {
      const px = pad.l + ((p.x - xmin) / (xmax - xmin)) * pw;
      const py = pad.t + ((yr[1] - p.y) / (yr[1] - yr[0])) * ph;
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    if (s.dots) {
      ctx.fillStyle = s.color;
      pts.forEach(p => {
        const px = pad.l + ((p.x - xmin) / (xmax - xmin)) * pw;
        const py = pad.t + ((yr[1] - p.y) / (yr[1] - yr[0])) * ph;
        ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
      });
    }
  });
}

function render(d) {
  document.getElementById("clockTxt").textContent = "更新于 " + d.server_time;
  const sel = document.getElementById("runSel");
  if (sel.options.length !== d.runs.length) {
    sel.innerHTML = d.runs.map(r => `<option value="${r}" ${r === d.run ? "selected" : ""}>${r}</option>`).join("");
  } else {
    sel.value = d.run;
  }

  const st = d.status || {};
  const step = st.step || (d.train.length ? d.train[d.train.length - 1].step : 0);
  const total = st.total_steps || 0;
  const pct = total > 0 ? Math.min(100, (100 * step / total)) : 0;
  const ep = st.epoch != null ? st.epoch + 1 : 1;
  document.getElementById("runName").textContent = d.run;
  document.getElementById("epochTxt").textContent = `｜ 第 ${ep} 轮 ｜ 参数 ${(st.params ? (st.params/1e6).toFixed(2)+'M' : '-')}`;
  const etaMin = st.eta_min != null ? (st.eta_min >= 60 ? (st.eta_min/60).toFixed(1)+' h' : st.eta_min.toFixed(0)+' min') : '-';
  document.getElementById("stepTxt").textContent = `Step ${step.toLocaleString()} / ${total.toLocaleString()} (${pct.toFixed(1)}%) ｜ ETA ${etaMin}`;
  document.getElementById("progBar").style.width = pct.toFixed(2) + "%";

  const lastTr = d.train.length ? d.train[d.train.length - 1] : null;
  const lastVa = d.val.length ? d.val[d.val.length - 1] : null;
  document.getElementById("kLoss").textContent = lastTr ? lastTr.loss.toFixed(4) : (st.loss ?? "-");
  document.getElementById("kGrad").textContent = lastTr ? `grad_norm ${lastTr.grad_norm}` : "等待首个 log...";
  document.getElementById("kBestVal").textContent = st.best_val != null ? st.best_val.toFixed(4) : "-";
  document.getElementById("kLastVal").textContent = lastVa ? (`step ${lastVa.step}: ${lastVa.val_loss.toFixed(4)}` + (lastVa.moe_summary ? " | MoE在线" : "")) : "等待首个 val...";
  const tps = st.tok_s || (lastTr ? lastTr.tok_s : 0);
  document.getElementById("kTps").textContent = tps ? (tps/1000).toFixed(1) + "k tok/s" : "-";
  const lr = st.lr || (lastTr ? lastTr.lr : 0);
  document.getElementById("kLr").textContent = lr ? `LR ${lr.toExponential(2)}` : "LR -";

  // GPU
  const g = d.gpu || {};
  const tempEl = document.getElementById("kTemp");
  if (g.ok) {
    tempEl.textContent = g.temp != null ? `${g.temp.toFixed(0)} °C` : "-";
    tempEl.className = "kpi-val " + (g.temp >= 80 ? "c-red" : g.temp >= 70 ? "c-orange" : "c-green");
    document.getElementById("kFan").textContent = `风扇 ${g.fan ?? '-'}% ｜ 频率 ${g.clock ?? '-'} MHz`;
    document.getElementById("kPower").textContent = g.power_draw != null ? `${g.power_draw.toFixed(0)} / ${(g.power_limit||0).toFixed(0)} W` : "-";
    document.getElementById("kUtil").textContent = `GPU ${g.util_gpu ?? '-'}% ｜ 显存带宽 ${g.util_mem ?? '-'}%`;
    document.getElementById("kVram").textContent = g.mem_used != null ? `${(g.mem_used/1024).toFixed(2)} / ${(g.mem_total/1024).toFixed(1)} GB` : "-";
    document.getElementById("kPeak").textContent = st.gpu_peak_mb ? `PyTorch 峰值 ${(st.gpu_peak_mb/1024).toFixed(2)} GB` : "-";
    document.getElementById("gpuNameTxt").textContent = g.name || "";
  }

  // 图表
  const epHalf = total > 0 ? total / 2 : null;
  drawChart("cLoss", [
    { color: "#58a6ff", width: 1.8, pts: d.train.map(p => ({ x: p.step, y: p.loss })) },
    { color: "#3fb950", width: 2.2, dots: true, pts: d.val.map(p => ({ x: p.step, y: p.val_loss })) },
  ], { vlineX: epHalf });

  const gh = d.gpu_hist || [];
  drawChart("cGpu", [
    { color: "#f85149", width: 2, axis: 0, pts: gh.map((p, i) => ({ x: i, y: p.temp })) },
    { color: "#bc8cff", width: 1.8, axis: 1, pts: gh.map((p, i) => ({ x: i, y: p.power_draw })) },
  ], {
    dual: true,
    fmt0: v => v.toFixed(0) + "°C",
    fmt1: v => v.toFixed(0) + "W",
    fmtX: i => gh[Math.min(gh.length - 1, Math.max(0, Math.round(i)))]?.t || ""
  });

  drawChart("cSpeed", [
    { color: "#39c5cf", width: 1.8, axis: 0, pts: d.train.map(p => ({ x: p.step, y: (p.tok_s || 0) / 1000 })) },
    { color: "#f0883e", width: 1.5, axis: 1, pts: d.train.map(p => ({ x: p.step, y: p.grad_norm })) },
  ], { dual: true, fmt0: v => v.toFixed(1) + "k", fmt1: v => v.toFixed(2), vlineX: epHalf });

  drawChart("cLr", [
    { color: "#d2a8ff", width: 2, pts: d.train.map(p => ({ x: p.step, y: (p.lr || 0) * 1e4 })) },
  ], { fmt0: v => v.toFixed(2) + "e-4", vlineX: epHalf });

  // 日志
  document.getElementById("logBox").textContent = (d.log_tail || []).join("\n") || "暂无日志";

  // 采样
  const smps = d.samples || [];
  const tabs = document.getElementById("sampleTabs");
  if (smps.length) {
    if (activeSampleIdx >= smps.length) activeSampleIdx = 0;
    tabs.innerHTML = smps.map((s, idx) =>
      `<span class="sample-tab ${idx === activeSampleIdx ? 'active' : ''}" onclick="pickSample(${idx})">${s.name.replace('.txt','')}</span>`
    ).join("");
    document.getElementById("sampleBox").textContent = smps[activeSampleIdx].text;
  } else {
    tabs.innerHTML = "";
    document.getElementById("sampleBox").textContent = "暂未生成采样（每 2000 步自动输出一次）";
  }
}

window.pickSample = (idx) => { activeSampleIdx = idx; fetchData(); };

async function fetchData() {
  try {
    const q = currentRun ? ("?run=" + encodeURIComponent(currentRun)) : "";
    const res = await fetch("/api/state" + q, { cache: "no-store" });
    const d = await res.json();
    if (d.ok) render(d);
  } catch (e) {
    console.error(e);
  }
}

fetchData();
setInterval(() => { if (!paused) fetchData(); }, 2500);
window.addEventListener("resize", () => fetchData());
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # 静默请求日志

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            runs = list_runs()
            req_run = (qs.get("run") or [""])[0]
            if not req_run or not os.path.isdir(os.path.join(OUT_ROOT, req_run)):
                req_run = DEFAULT_RUN if os.path.isdir(os.path.join(OUT_ROOT, DEFAULT_RUN)) else (runs[0] if runs else DEFAULT_RUN)
            data = read_run_state(req_run)
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global DEFAULT_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--run", default=DEFAULT_RUN)
    args = ap.parse_args()
    DEFAULT_RUN = args.run

    t = threading.Thread(target=gpu_poller, daemon=True)
    t.start()
    time.sleep(0.3)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"监控台已启动：http://{args.host}:{args.port}  (默认任务: {DEFAULT_RUN})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
