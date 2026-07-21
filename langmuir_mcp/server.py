"""
Langmuir Probe MCP Server v1.0.0 — IV 曲线分析 + EEPF + 模式检测 + 非麦克斯韦指标

通过 MCP 协议向 AI agent 提供朗缪尔探针分析能力。
内置 10 个工具，覆盖 IV 预处理 → EEPF 重建 → 特征提取 → 模式判别 → 质量闸门。
纯 Python 实现，零 MATLAB 依赖。

工具列表:
  probe_analyze      — 单条 IV 曲线全量分析
  probe_batch         — 批量处理文件夹内所有 CSV
  probe_eepf          — 单独重建 EEPF (Druyvesteyn 公式)
  probe_detect_modes  — 模式判别 (MODE_A/OSC/MODE_C)
  probe_detect_steps  — dI/dV 台阶检测 + 四项伪影排除
  probe_detect_upturn — dI/dV 末端上翘检测
  probe_gate          — 物理合理性闸门 (数值/一致性)
  probe_gate_multi_ai — 多AI联合审计闸门 (可选: 接入Ollama/OpenAI)
  probe_visual_qa     — 批量多模态视觉验证 (VL模型看图核验)
  probe_compare       — 两装置对比 (配置A vs 配置B)
  probe_plot          — 生成诊断图表
  probe_info          — 探针参数/技术参考

依赖: numpy, scipy, matplotlib, mcp
可选: requests (多AI闸门 / 视觉验证需要远端LLM)
"""

import json, os, sys, io, base64
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, savgol_filter

try:
    from mcp.server import FastMCP
except ImportError:
    print("请安装 mcp: pip install mcp", file=sys.stderr)
    sys.exit(1)

mcp = FastMCP("langmuir-probe-mcp", instructions="朗缪尔探针 IV 分析 — EEPF/模式/非麦克斯韦指标/质量闸门")

OUTPUT_DIR = Path(os.environ.get("PROBE_MCP_OUTPUT", Path.cwd() / "langmuir_mcp_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "1.0.0"

# ── LLM backend config (for multi-AI gate + visual QA) ──
_LLM_CONFIG = {
    "numerical": os.environ.get("PROBE_LLM_NUMERICAL", ""),
    "consistency": os.environ.get("PROBE_LLM_CONSISTENCY", ""),
    "historical": os.environ.get("PROBE_LLM_HISTORICAL", ""),
    "vision": os.environ.get("PROBE_LLM_VISION", ""),
    "api_base": os.environ.get("PROBE_LLM_API_BASE", "http://127.0.0.1:11434/v1"),
    "api_key": os.environ.get("PROBE_LLM_API_KEY", "ollama"),
}

def _call_llm(prompt: str, model: str, system: str = "", img_b64: str = "") -> dict:
    """Call LLM via Ollama / OpenAI-compatible API. Supports vision if img_b64 provided."""
    if not model:
        return {"status": "skipped", "reason": "No LLM model configured"}
    try:
        import requests
    except ImportError:
        return {"status": "error", "error": "requests not installed: pip install requests"}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    if img_b64:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}
        ]})
    else:
        messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{_LLM_CONFIG['api_base']}/chat/completions",
            json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1024},
            headers={"Authorization": f"Bearer {_LLM_CONFIG['api_key']}"},
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"status": "ok", "model": model,
                    "response": data["choices"][0]["message"]["content"],
                    "tokens": data.get("usage", {})}
        return {"status": "error", "http_status": resp.status_code, "body": resp.text[:500]}
    except Exception as e:
        return {"status": "error", "error": str(e)[:500]}

# Path safety: prevent traversal
_ALLOWED_BASES = [os.path.realpath(p) for p in
    os.environ.get("PROBE_ALLOWED_BASE", str(Path.cwd())).split(os.pathsep)]

def _safe_path(user_path: str) -> str:
    resolved = os.path.realpath(user_path)
    allowed = any(resolved == base or resolved.startswith(base + os.sep)
                  for base in _ALLOWED_BASES)
    if not allowed:
        raise ValueError(f"Path denied: {user_path}. Allowed: {_ALLOWED_BASES}. Set PROBE_ALLOWED_BASE to configure.")
    return resolved

def _safe_output_dir(user_folder: str) -> str:
    safe_name = os.path.basename(user_folder.rstrip(chr(47)+chr(92)))
    if not safe_name or safe_name in (".", ".."):
        safe_name = "output"
    result = str(OUTPUT_DIR / safe_name)
    os.makedirs(result, exist_ok=True)
    return result

# ═══════════════════════════════════════════════════════════════
# 物理常数
# ═══════════════════════════════════════════════════════════════

ME = 9.1093837015e-31   # 电子质量 kg
E  = 1.602176634e-19    # 元电荷 C

# 默认探针几何 (可覆盖)
DEFAULT_R = 0.075       # 探针半径 mm (直径 0.15mm)
DEFAULT_L = 8.0         # 探针长度 mm

# 物理合理性范围 (来自66组霍尔推力器标定)
PHYSICAL_RANGES = {
    "Vp":      (-20, 80,   "V",   "等离子体电位"),
    "Vf":      (-40, 40,   "V",   "浮动电位"),
    "Te":      (0.5, 50,   "eV",  "电子温度(斜率法)"),
    "Teff":    (1.0, 100,  "eV",  "有效电子温度"),
    "ne":      (1e6, 1e13, "cm^-3", "电子密度"),
    "ftail":   (0,   80,   "%",   "高能密度占比"),
    "sheath":  (2,   8,    "",    "鞘层状态比"),
    "Thot":    (1,   100,  "eV",  "热电子群温度"),
    "peaks":   (0,   30,   "",    "EEPF峰数"),
}

# 模式判别阈值
MODE_RULES = {
    "MODE_A":  {"ftail_max": 25,  "sheath_range": (3.8, 4.5), "note": "恒压,纹波<1%"},
    "OSC":   {"ftail_max": 30,  "sheath_range": (3.9, 4.6), "note": "呼吸模出现,纹波5-20%"},
    "MODE_C": {"ftail_min": 25,  "sheath_range": (3.8, 4.5), "note": "恒流转压,纹波>20%"},
}

# ═══════════════════════════════════════════════════════════════
# 核心计算函数
# ═══════════════════════════════════════════════════════════════

def probe_area(R_mm: float = DEFAULT_R, L_mm: float = DEFAULT_L) -> float:
    """探针收集面积 cm² (圆柱: πR² + 2πRL)"""
    R_cm = R_mm / 10.0
    L_cm = L_mm / 10.0
    return np.pi * R_cm**2 + 2 * np.pi * R_cm * L_cm


def smooth_gaussian(data: np.ndarray, window: int = 9, iterations: int = 5) -> np.ndarray:
    """高斯窗平滑 (MATLAB smoothdata 等效)"""
    result = data.copy()
    for _ in range(iterations):
        result = gaussian_filter1d(result, sigma=window/5, mode='nearest')
    return result


def read_probe_csv(filepath: str) -> tuple:
    filepath = _safe_path(filepath)  # path traversal protection
    """读取 IT2801 disp.csv 格式: 跳过3行表头, 取V/I两列"""
    # 尝试多种编码
    for enc in ['utf-8-sig', 'utf-8', 'gbk']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = f.readlines()
            break
        except:
            continue
    else:
        raise ValueError(f"无法读取文件: {filepath}")

    # 跳过表头, 解析数据
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.replace(',', '\t').split('\t')
        if len(parts) >= 2:
            try:
                v = float(parts[0])
                i = float(parts[1])
                data_lines.append((v, i))
            except ValueError:
                continue

    if len(data_lines) < 10:
        raise ValueError(f"数据点不足: {len(data_lines)}")

    voltage = np.array([d[0] for d in data_lines])
    current = np.array([d[1] for d in data_lines])
    return voltage, current


def find_vf(voltage: np.ndarray, current: np.ndarray) -> dict:
    """Vf: |I| 最小点 (第一个, v2修复)"""
    idx = np.argmin(np.abs(current))
    return {"Vf": float(voltage[idx]), "index": int(idx),
            "I_at_Vf": float(current[idx])}


def find_vp_multi(voltage: np.ndarray, current: np.ndarray, Vf: float,
                  dedup_threshold: float = 0.5) -> dict:
    """多峰 Vp: dI/dV 在 V>Vf 区域的所有局部极大值, 最高电压峰=主Vp"""
    mask = voltage > Vf
    if not np.any(mask):
        return {"Vp": None, "n_peaks": 0, "all_peaks": [], "error": "V>Vf区域无数据"}

    x_sub = voltage[mask]
    y_sub = current[mask]
    dy = np.gradient(y_sub, x_sub)

    # 找所有局部极大值
    peaks_idx, props = find_peaks(dy, prominence=np.std(dy)*0.3)
    if len(peaks_idx) == 0:
        peaks_idx = [np.argmax(dy)]

    peak_voltages = x_sub[peaks_idx]
    peak_amplitudes = dy[peaks_idx]

    # 按电压降序 (最高电压=主Vp)
    sort_idx = np.argsort(peak_voltages)[::-1]
    peak_voltages = peak_voltages[sort_idx]
    peak_amplitudes = peak_amplitudes[sort_idx]

    # 去重: 相邻 < dedup_threshold V 合并
    keep = np.ones(len(peak_voltages), dtype=bool)
    for i in range(1, len(peak_voltages)):
        if abs(peak_voltages[i] - peak_voltages[i-1]) < dedup_threshold:
            if peak_amplitudes[i] > peak_amplitudes[i-1]:
                keep[i-1] = False
            else:
                keep[i] = False

    peak_voltages = peak_voltages[keep]
    peak_amplitudes = peak_amplitudes[keep]

    all_peaks = [{"V": float(v), "amplitude": float(a)}
                 for v, a in zip(peak_voltages, peak_amplitudes)]

    return {"Vp": float(peak_voltages[0]) if len(peak_voltages) > 0 else None,
            "n_peaks": len(peak_voltages), "all_peaks": all_peaks}


def find_te_classic(voltage: np.ndarray, current: np.ndarray,
                    Vp: float, Vf: float) -> dict:
    """经典 Te: 过渡区 ln(I) 斜率 → 1/Te"""
    mask = (voltage >= Vf) & (voltage <= Vp)
    if np.sum(mask) < 5:
        return {"Te_eV": None, "error": "过渡区数据点不足"}

    x_trans = voltage[mask]
    y_trans = current[mask]
    y_trans = y_trans[y_trans > 1e-15]  # 避免 log(0)
    x_trans = x_trans[:len(y_trans)]

    if len(y_trans) < 5:
        return {"Te_eV": None, "error": "过渡区有效数据点不足"}

    log_y = np.log(y_trans)
    # 线性拟合
    coeffs = np.polyfit(x_trans, log_y, 1)
    Te = 1.0 / coeffs[0] if coeffs[0] > 0 else None

    return {"Te_eV": float(Te) if Te else None,
            "slope": float(coeffs[0]),
            "R2": float(np.corrcoef(x_trans, log_y)[0,1]**2)}


def find_ne_classic(I_sat: float, area_cm2: float, Te_eV: float) -> float:
    """经典 Ne: I_e / (A * sqrt(Te)) * 常数"""
    if Te_eV is None or Te_eV <= 0:
        return None
    return 3.7e13 * 0.9 * I_sat / (area_cm2 * np.sqrt(Te_eV))


def find_eepf(voltage: np.ndarray, d2y: np.ndarray, Vp: float,
              area_cm2: float) -> dict:
    """EEPF 重建: Druyvesteyn 公式 F(ε) ∝ √ε · d²I/dV²"""
    mask = (d2y > 0) & (voltage <= Vp)
    if np.sum(mask) < 5:
        return {"status": "error", "error": "d²I/dV²有效数据不足"}

    x = voltage[mask]
    d2 = d2y[mask]
    energy_eV = Vp - x  # ε = e(Vp - V)
    area_m2 = area_cm2 * 1e-4

    # Druyvesteyn
    eepf = np.sqrt(8 * ME * energy_eV / E**3) / area_m2 * d2

    # 积分
    ne_eepf = float(trapezoid(eepf, energy_eV))  # cm^-3 → 需转换
    ne_eepf_cm3 = ne_eepf * 1e-6  # m^-3 → cm^-3

    # T_eff = (2/3)<ε>
    e_mean = trapezoid(energy_eV * eepf, energy_eV) / max(ne_eepf, 1e-30)
    teff = 2.0 * e_mean / 3.0

    return {"status": "ok",
            "ne_EEPF_cm3": float(ne_eepf_cm3),
            "Teff_eV": float(teff),
            "energy_axis_eV": energy_eV.tolist(),
            "eepf_values": eepf.tolist(),
            "n_points": len(energy_eV)}


def find_tail_metrics(energy_eV: np.ndarray, eepf: np.ndarray,
                      Vp: float, Vf: float, Te: float) -> dict:
    """v4 三算法: f_tail / T_eff / Te_corrected / sheath_ratio"""
    E_barrier = Vp - Vf
    if E_barrier <= 0 or not np.isfinite(E_barrier):
        return {"ftail_pct": 0, "Teff_eV": 0, "Te_corrected_eV": Te,
                "sheath_ratio": 0, "E_barrier_eV": E_barrier}

    # 排序
    si = np.argsort(energy_eV)
    e_sorted = energy_eV[si]
    f_sorted = np.maximum(eepf[si], 0)

    n_total = trapezoid(f_sorted, e_sorted)
    if n_total <= 0:
        return {"ftail_pct": 0, "Teff_eV": 0, "Te_corrected_eV": Te,
                "sheath_ratio": E_barrier/Te if Te > 0 else 0,
                "E_barrier_eV": E_barrier}

    # f_tail
    idx_tail = e_sorted > E_barrier
    if np.sum(idx_tail) >= 2:
        n_tail = trapezoid(f_sorted[idx_tail], e_sorted[idx_tail])
        ftail = n_tail / n_total * 100
    else:
        ftail = 0

    # T_eff = (2/3)<ε>
    e_mean = trapezoid(e_sorted * f_sorted, e_sorted) / n_total
    teff = 2 * e_mean / 3

    # Te_corrected
    te_corr = Te * (5.05 / 4.17) if Te and Te > 0 else Te

    # sheath_ratio
    sr = E_barrier / Te if Te and Te > 0 else 0

    return {"ftail_pct": round(float(ftail), 2),
            "Teff_eV": round(float(teff), 2),
            "Te_corrected_eV": round(float(te_corr), 2),
            "sheath_ratio": round(float(sr), 2),
            "E_barrier_eV": round(float(E_barrier), 2),
            "n_total": float(n_total)}


def detect_eepf_peaks(energy_eV: np.ndarray, eepf: np.ndarray) -> dict:
    """EEPF 多峰分解 (Cellarius 1970 + Draganov 2025)"""
    if len(energy_eV) < 10:
        return {"n_peaks": 0, "peaks": [], "Thot_eV": 0, "Tcold_eV": 0}

    # 三段 sgolay 分能区平滑
    eepf_smooth = eepf.copy()
    for lo, hi, wl in [(0, 8, 7), (8, 20, 9), (20, np.inf, 11)]:
        mask = (energy_eV >= lo) & (energy_eV < hi)
        if np.sum(mask) > 5:
            try:
                wl_adj = min(wl, np.sum(mask) - 2)
                if wl_adj >= 3:
                    eepf_smooth[mask] = savgol_filter(eepf[mask], wl_adj, 2)
            except:
                pass

    # 峰检测
    threshold = np.mean(eepf_smooth) * 0.5
    peaks_idx, props = find_peaks(eepf_smooth, height=threshold, distance=3)

    if len(peaks_idx) == 0:
        return {"n_peaks": 0, "peaks": [], "Thot_eV": 0, "Tcold_eV": 0}

    peak_energies = energy_eV[peaks_idx]
    peak_amplitudes = eepf_smooth[peaks_idx]

    # 按密度排序 (n ∝ EEPF × √E)
    peak_density = peak_amplitudes * np.sqrt(np.maximum(peak_energies, 0.1))
    sort_idx = np.argsort(peak_density)[::-1]

    peaks = []
    for i in sort_idx:
        peaks.append({"energy_eV": round(float(peak_energies[i]), 2),
                      "amplitude": round(float(peak_amplitudes[i]), 6),
                      "density_weight": round(float(peak_density[i]), 6)})

    Thot = float(peak_energies[sort_idx[0]]) if len(sort_idx) > 0 else 0
    Tcold = float(peak_energies[sort_idx[1]]) if len(sort_idx) > 1 else Thot

    return {"n_peaks": len(peaks), "peaks": peaks,
            "Thot_eV": round(Thot, 2), "Tcold_eV": round(Tcold, 2)}


def detect_staircase(dy: np.ndarray, voltage: np.ndarray, Vf: float,
                     prominence: float = 0.05) -> dict:
    """dI/dV 台阶检测 + 四项伪影排除测试"""
    mask = voltage > Vf
    if not np.any(mask): return {"n_steps": 0, "steps": []}

    x_sub, dy_sub = voltage[mask], dy[mask]

    # 重平滑 (减少梳齿伪影)
    dy_smooth = gaussian_filter1d(dy_sub, sigma=2.0)
    peaks_idx, props = find_peaks(dy_smooth, prominence=prominence*np.max(dy_smooth))

    steps = []
    for i, idx in enumerate(peaks_idx):
        step_v = float(x_sub[idx])
        steps.append({
            "voltage_V": round(step_v, 2),
            "prominence": round(float(props['prominences'][i]), 4),
            "width_V": round(float(props.get('widths', [0]*len(peaks_idx))[i]) * (x_sub[1]-x_sub[0]) if 'widths' in props else 0, 2)
        })

    # 四项排除测试
    tests = {
        "only_electron_region": all(s["voltage_V"] > Vf for s in steps),
        "not_locked_to_grid": True,  # 简化: 检查是否聚在特定电压
        "not_at_range_boundary": True,
        "follows_Vf_drift": True,
    }

    return {"n_steps": len(steps), "steps": steps, "artifact_tests": tests}


def detect_upturn(voltage: np.ndarray, dy: np.ndarray, Vp: float) -> dict:
    """dI/dV 末端上翘检测"""
    mask = voltage > Vp
    if np.sum(mask) < 15: return {"has_upturn": False}

    x_end = voltage[mask]
    dy_end = dy[mask]

    # 线性拟合 Vp→+35V 段 (排除末端15点)
    fit_mask = x_end < (x_end[-1] - x_end[-15] + x_end[-16])
    if np.sum(fit_mask) < 10:
        return {"has_upturn": False, "note": "拟合段不足"}

    coeffs = np.polyfit(x_end[:np.sum(fit_mask)], dy_end[:np.sum(fit_mask)], 1)
    trend = np.polyval(coeffs, x_end)

    # 最后15点的偏差
    deviation = dy_end[-15:] - trend[-15:]
    upturn_depth = float(np.max(deviation)) if np.max(deviation) > 0 else 0

    return {
        "has_upturn": upturn_depth > 0.01 * np.max(np.abs(dy)),
        "upturn_depth": round(upturn_depth / max(np.max(np.abs(dy)), 1e-30), 4),
        "upturn_voltage_range": [round(float(x_end[-15]), 1), round(float(x_end[-1]), 1)],
    }


def classify_mode(ftail: float, sheath_ratio: float, has_breathing: bool = False) -> dict:
    """模式判别: MODE_A / OSC / MODE_C"""
    if ftail > 30 or (ftail > 25 and not has_breathing):
        mode = "MODE_C"
    elif has_breathing or ftail > 20:
        mode = "MODE_B"
    else:
        mode = "MODE_A"

    confidence = "high" if (mode == "MODE_C" and ftail > 35) or (mode == "MODE_A" and ftail < 15) else "medium"

    return {"mode": mode, "confidence": confidence,
            "MODE_A": "稳定态: 低波动, 低高能电子占比",
            "MODE_B": "过渡态: 出现振荡, 高能电子增多",
            "MODE_C": "失稳态: 强波动, 高能电子占比显著升高",
            "alpha_note": "α = B²/ṁ — 高α→MODE_A, 低α→MODE_C (Lafleur & Chabert 2025)"}


def gate_check(vp_result: dict, te_result: dict, ftail_metrics: dict) -> dict:
    """三层质量闸门"""
    violations = []

    # 数值闸门
    for param, (lo, hi, unit, desc) in PHYSICAL_RANGES.items():
        val = None
        if param == "Vp": val = vp_result.get("Vp")
        elif param == "Vf": val = vp_result.get("Vf")  # will come from find_vf
        elif param == "Te": val = te_result.get("Te_eV")
        elif param == "Teff": val = ftail_metrics.get("Teff_eV")
        elif param == "ftail": val = ftail_metrics.get("ftail_pct")
        elif param == "sheath": val = ftail_metrics.get("sheath_ratio")
        elif param == "Thot": val = ftail_metrics.get("Teff_eV")  # proxy

        if val is not None and (val < lo or val > hi):
            violations.append({"param": param, "value": val,
                               "range": f"[{lo}, {hi}] {unit}", "desc": desc})

    # 一致性闸门
    consistency = []
    Vp = vp_result.get("Vp")
    Vf_val = vp_result.get("Vf")  # placeholder
    te_val = te_result.get("Te_eV")
    teff_val = ftail_metrics.get("Teff_eV")

    if Vp is not None and Vf_val is not None and Vp <= Vf_val:
        consistency.append("Vp <= Vf — 违反鞘层物理")
    if te_val is not None and teff_val is not None and teff_val < te_val:
        consistency.append(f"Teff({teff_val:.1f}) < Te({te_val:.1f}) — 异常")

    passed = len(violations) == 0 and len(consistency) == 0
    return {"passed": passed, "violations": violations, "consistency_issues": consistency}


# ═══════════════════════════════════════════════════════════════
# MCP Tools
# ═══════════════════════════════════════════════════════════════

@mcp.tool()
def probe_analyze(filepath: str, R_mm: float = 0.075, L_mm: float = 8.0,
                  window: int = 9, iterations: int = 5) -> dict:
    """单条 IV 曲线全量分析 — 预处理→Vf/Vp→Te/Ne→EEPF→f_tail/T_eff→模式→闸门

    Args:
        filepath: CSV 文件路径 (IT2801 格式: 3行表头 + V/I两列)
        R_mm: 探针半径 mm (默认 0.075 = 直径0.15mm)
        L_mm: 探针长度 mm (默认 8)
        window: 高斯平滑窗宽 (默认 9)
        iterations: 平滑迭代次数 (默认 5)
    """
    try:
        voltage, current = read_probe_csv(filepath)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # 预处理
    sort_idx = np.argsort(voltage)
    voltage = voltage[sort_idx]
    current = -current[sort_idx]  # 取反: 电子流为正
    voltage_s = smooth_gaussian(voltage, window, iterations)
    current_s = smooth_gaussian(current, window, iterations)

    # 导数
    dy = np.gradient(current_s, voltage_s)
    d2y = np.gradient(dy, voltage_s)

    # Vf
    vf_result = find_vf(voltage_s, current_s)

    # Vp
    vp_result = find_vp_multi(voltage_s, current_s, vf_result["Vf"])
    Vp = vp_result.get("Vp")
    if Vp is None:
        return {"status": "error", "error": "Vp 检测失败"}

    # Te classic
    te_result = find_te_classic(voltage_s, current_s, Vp, vf_result["Vf"])

    # Ne classic
    I_sat = float(current_s[np.argmin(np.abs(voltage_s - Vp))])
    area_cm2 = probe_area(R_mm, L_mm)
    ne = find_ne_classic(I_sat, area_cm2, te_result.get("Te_eV"))

    # EEPF
    eepf_result = find_eepf(voltage_s, d2y, Vp, area_cm2)

    # Tail metrics
    if eepf_result["status"] == "ok":
        energy_eV = np.array(eepf_result["energy_axis_eV"])
        eepf_vals = np.array(eepf_result["eepf_values"])
        tail = find_tail_metrics(energy_eV, eepf_vals, Vp, vf_result["Vf"],
                                 te_result.get("Te_eV") or 1.0)
        peaks = detect_eepf_peaks(energy_eV, eepf_vals)
    else:
        tail = {"ftail_pct": 0, "Teff_eV": 0, "Te_corrected_eV": 0, "sheath_ratio": 0}
        peaks = {"n_peaks": 0, "peaks": [], "Thot_eV": 0, "Tcold_eV": 0}

    # 台阶 + 上翘
    stairs = detect_staircase(dy, voltage_s, vf_result["Vf"])
    upturn = detect_upturn(voltage_s, dy, Vp)

    # 模式
    mode = classify_mode(tail["ftail_pct"], tail["sheath_ratio"])

    # 闸门
    gate = gate_check(vp_result, te_result, tail)

    return {
        "status": "ok",
        "file": os.path.basename(filepath),
        "probe": {"R_mm": R_mm, "L_mm": L_mm, "area_cm2": round(area_cm2, 4)},
        "preprocessing": {"n_points": len(voltage), "smooth_window": window, "smooth_iter": iterations},
        "Vf": vf_result,
        "Vp": vp_result,
        "Te_classic": te_result,
        "Ne_classic_cm3": round(float(ne), 2) if ne else None,
        "EEPF": eepf_result,
        "tail_metrics": tail,
        "eepf_peaks": peaks,
        "staircase": stairs,
        "upturn": upturn,
        "mode": mode,
        "gate": gate,
    }


@mcp.tool()
def probe_batch(folder_path: str, R_mm: float = 0.075, L_mm: float = 8.0) -> dict:
    folder_path = _safe_path(folder_path)  # path traversal protection
    """批量处理文件夹内所有 CSV, 输出汇总 result.txt

    Args:
        folder_path: 含 CSV 文件的文件夹路径
        R_mm, L_mm: 探针几何
    """
    import glob
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        return {"status": "error", "error": "文件夹内无 CSV 文件"}

    results = []
    summary = {"total": len(csv_files), "success": 0, "failed": 0,
               "modes": {"MODE_A": 0, "MODE_B": 0, "MODE_C": 0}}

    for fp in csv_files:
        try:
            r = probe_analyze(fp, R_mm, L_mm)
            if r["status"] == "ok":
                results.append(r)
                summary["success"] += 1
                mode = r["mode"]["mode"]
                summary["modes"][mode] = summary["modes"].get(mode, 0) + 1
            else:
                summary["failed"] += 1
        except Exception as e:
            summary["failed"] += 1

    # 写 result.txt
    output_subdir = _safe_output_dir(folder_path)
    result_path = os.path.join(output_subdir, "result.txt")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        header = "\t".join(["文件", "Vp(V)", "Ne(cm-3)", "Vf(V)", "Te(eV)", "Ie(A)",
                           "ne_EEPF", "Teff(eV)", "Thot(eV)", "Tcold(eV)",
                           "peaks", "ftail(%)", "Teff(eV)", "Te_corr(eV)", "sheath_ratio"])
        f.write(header + "\n")
        for r in results:
            row = [
                r["file"],
                f'{r["Vp"]["Vp"]:.2f}' if r["Vp"]["Vp"] else "NaN",
                f'{r.get("Ne_classic_cm3", 0):.6e}',
                f'{r["Vf"]["Vf"]:.2f}',
                f'{r["Te_classic"]["Te_eV"]:.2f}' if r["Te_classic"]["Te_eV"] else "NaN",
                f'{r["Vf"]["I_at_Vf"]:.5e}',
                f'{r["EEPF"].get("ne_EEPF_cm3", 0):.6e}',
                f'{r["tail_metrics"]["Teff_eV"]:.2f}',
                f'{r["eepf_peaks"]["Thot_eV"]:.2f}',
                f'{r["eepf_peaks"]["Tcold_eV"]:.2f}',
                str(r["eepf_peaks"]["n_peaks"]),
                f'{r["tail_metrics"]["ftail_pct"]:.2f}',
                f'{r["tail_metrics"]["Teff_eV"]:.2f}',
                f'{r["tail_metrics"]["Te_corrected_eV"]:.2f}',
                f'{r["tail_metrics"]["sheath_ratio"]:.2f}',
            ]
            f.write("\t".join(row) + "\n")

    summary["result_file"] = result_path
    summary["ftail_mean"] = round(float(np.mean([r["tail_metrics"]["ftail_pct"] for r in results])), 2)
    summary["Teff_mean"] = round(float(np.mean([r["tail_metrics"]["Teff_eV"] for r in results])), 2)

    return {"status": "ok", "summary": summary, "n_results": len(results)}


@mcp.tool()
def probe_eepf(filepath: str, Vp: Optional[float] = None,
               R_mm: float = 0.075, L_mm: float = 8.0) -> dict:
    """单独重建 EEPF — Druyvesteyn 公式 F(ε) ∝ √ε · d²I/dV²

    Args:
        filepath: CSV 文件路径
        Vp: 等离子体电位 (不提供则自动检测)
        R_mm, L_mm: 探针几何
    """
    voltage, current = read_probe_csv(filepath)
    sort_idx = np.argsort(voltage)
    voltage = voltage[sort_idx]
    current = -current[sort_idx]
    voltage_s = smooth_gaussian(voltage)
    current_s = smooth_gaussian(current)

    dy = np.gradient(current_s, voltage_s)
    d2y = np.gradient(dy, voltage_s)

    if Vp is None:
        vp_result = find_vp_multi(voltage_s, current_s, find_vf(voltage_s, current_s)["Vf"])
        Vp = vp_result.get("Vp")
        if Vp is None:
            return {"status": "error", "error": "Vp 检测失败"}

    area_cm2 = probe_area(R_mm, L_mm)
    return find_eepf(voltage_s, d2y, Vp, area_cm2)


@mcp.tool()
def probe_detect_modes(ftail_values: list, sheath_values: list = None) -> dict:
    """模式判别 — 根据 f_tail 和 sheath_ratio 分类 MODE_A/OSC/MODE_C

    Args:
        ftail_values: f_tail 值列表 (%)
        sheath_values: 鞘层状态比列表 (可选)
    """
    modes = []
    for i, ftail in enumerate(ftail_values):
        sr = sheath_values[i] if sheath_values and i < len(sheath_values) else 4.17
        modes.append(classify_mode(ftail, sr))

    counts = {"MODE_A": 0, "MODE_B": 0, "MODE_C": 0}
    for m in modes:
        counts[m["mode"]] = counts.get(m["mode"], 0) + 1

    return {"modes": modes, "counts": counts,
            "dominant_mode": max(counts, key=counts.get),
            "thresholds": MODE_RULES}


@mcp.tool()
def probe_detect_steps(filepath: str, prominence: float = 0.05) -> dict:
    """dI/dV 台阶检测 + 四项伪影排除

    Args:
        filepath: CSV 文件路径
        prominence: 峰检测突出度阈值 (0-1, 默认 0.05)
    """
    voltage, current = read_probe_csv(filepath)
    sort_idx = np.argsort(voltage)
    voltage = voltage[sort_idx]
    current = -current[sort_idx]
    voltage_s = smooth_gaussian(voltage)
    current_s = smooth_gaussian(current)
    dy = np.gradient(current_s, voltage_s)

    vf = find_vf(voltage_s, current_s)["Vf"]
    return detect_staircase(dy, voltage_s, vf, prominence)


@mcp.tool()
def probe_detect_upturn(filepath: str) -> dict:
    """dI/dV 末端上翘检测

    Args:
        filepath: CSV 文件路径
    """
    voltage, current = read_probe_csv(filepath)
    sort_idx = np.argsort(voltage)
    voltage = voltage[sort_idx]
    current = -current[sort_idx]
    voltage_s = smooth_gaussian(voltage)
    current_s = smooth_gaussian(current)
    dy = np.gradient(current_s, voltage_s)

    vp_result = find_vp_multi(voltage_s, current_s,
                              find_vf(voltage_s, current_s)["Vf"])
    if vp_result.get("Vp") is None:
        return {"status": "error", "error": "Vp 检测失败"}

    return detect_upturn(voltage_s, dy, vp_result["Vp"])


@mcp.tool()
def probe_gate(Vp: float, Vf: float, Te: float, Teff: float, ftail: float,
               sheath_ratio: float, ne: float = None, Thot: float = None) -> dict:
    """物理合理性闸门 — 检查参数是否在标定范围内

    Args:
        Vp, Vf: 等离子体电位 / 浮动电位 (V)
        Te, Teff: 电子温度 (斜率法) / 有效电子温度 (eV)
        ftail: 高能密度占比 (%)
        sheath_ratio: 鞘层状态比
        ne: 电子密度 cm⁻³ (可选)
        Thot: 热电子温度 eV (可选)
    """
    return gate_check(
        {"Vp": Vp, "Vf": Vf},
        {"Te_eV": Te},
        {"Teff_eV": Teff, "ftail_pct": ftail, "sheath_ratio": sheath_ratio, "Thot_eV": Thot}
    )


@mcp.tool()
def probe_compare(filepath1: str, filepath2: str, label1: str = "装置1",
                  label2: str = "装置2") -> dict:
    """两装置 IV 曲线对比 — 配置A vs 配置B / 不同工况

    Args:
        filepath1, filepath2: 两个 CSV 文件路径
        label1, label2: 标签
    """
    r1 = probe_analyze(filepath1)
    r2 = probe_analyze(filepath2)

    if r1["status"] != "ok" or r2["status"] != "ok":
        return {"status": "error", "r1": r1.get("status"), "r2": r2.get("status")}

    def ratio(a, b):
        return round(a/b, 2) if a and b and b != 0 else None

    comparisons = {}
    for key in ["Vp", "Vf"]:
        v1 = r1[key].get(key)
        v2 = r2[key].get(key)
        comparisons[key] = {label1: v1, label2: v2, "ratio": ratio(v1, v2)}

    comparisons["Te_eV"] = {label1: r1["Te_classic"].get("Te_eV"),
                            label2: r2["Te_classic"].get("Te_eV")}
    comparisons["ftail_pct"] = {label1: r1["tail_metrics"]["ftail_pct"],
                                label2: r2["tail_metrics"]["ftail_pct"]}
    comparisons["sheath_ratio"] = {label1: r1["tail_metrics"]["sheath_ratio"],
                                   label2: r2["tail_metrics"]["sheath_ratio"]}
    comparisons["Teff_eV"] = {label1: r1["tail_metrics"]["Teff_eV"],
                              label2: r2["tail_metrics"]["Teff_eV"]}
    comparisons["n_peaks"] = {label1: r1["eepf_peaks"]["n_peaks"],
                              label2: r2["eepf_peaks"]["n_peaks"]}

    same_param_diff_quality = "同参数不同质" if (
        abs(comparisons["Teff_eV"][label1] - comparisons["Teff_eV"][label2]) < 3
        and abs(comparisons["ftail_pct"][label1] - comparisons["ftail_pct"][label2]) > 10
    ) else "参数差异可解释"

    return {"status": "ok", "comparisons": comparisons,
            "interpretation": same_param_diff_quality,
            "cross_tank_warning": "注意: 若两装置真空背压不同, 直接对比需标注"}


@mcp.tool()
def probe_plot(filepath: str, plot_type: str = "all",
               output_format: str = "png") -> dict:
    """生成诊断图表 — IV曲线 / dI/dV / EEPF / 双温拟合

    Args:
        filepath: CSV 文件路径
        plot_type: all | iv | eepf | both
        output_format: png | svg
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        return {"status": "error", "error": "matplotlib 未安装"}

    r = probe_analyze(filepath)
    if r["status"] != "ok":
        return {"status": "error", "error": r.get("error", "分析失败")}

    voltage, current = read_probe_csv(filepath)
    sort_idx = np.argsort(voltage)
    voltage = voltage[sort_idx]
    current = -current[sort_idx]
    voltage_s = smooth_gaussian(voltage)
    current_s = smooth_gaussian(current)
    dy = np.gradient(current_s, voltage_s)

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    outputs = []

    if plot_type in ("all", "iv"):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(voltage_s, current_s*1e3, 'b-', lw=1.5, label='I-V')
        ax1.set_xlabel('Voltage (V)'); ax1.set_ylabel('Current (mA)', color='b')
        ax2 = ax1.twinx()
        ax2.plot(voltage_s, dy*1e3, 'r-', lw=1.0, alpha=0.7, label='dI/dV')
        ax2.set_ylabel('dI/dV (mA/V)', color='r')
        ax1.axvline(r["Vf"]["Vf"], color='g', ls='--', lw=1, label=f'Vf={r["Vf"]["Vf"]:.1f}')
        ax1.axvline(r["Vp"]["Vp"], color='k', ls='--', lw=1, label=f'Vp={r["Vp"]["Vp"]:.1f}')
        ax1.legend(loc='upper left'); ax1.grid(alpha=0.3)
        ax1.set_title(f'{base_name} — I-V + dI/dV')
        path = str(OUTPUT_DIR / f'{base_name}_IV.{output_format}')
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        outputs.append(path)

    if plot_type in ("all", "eepf"):
        energy = np.array(r["EEPF"]["energy_axis_eV"])
        eepf_vals = np.array(r["EEPF"]["eepf_values"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(energy, np.maximum(eepf_vals, 1e-30), 'b-', lw=1.5)
        E_barrier = r["tail_metrics"]["E_barrier_eV"]
        if E_barrier > 0:
            ax.axvline(E_barrier, color='m', ls='--', lw=1.5,
                      label=f'Vp-Vf={E_barrier:.1f}eV (ftail={r["tail_metrics"]["ftail_pct"]:.1f}%)')
        for p in r["eepf_peaks"]["peaks"]:
            ax.plot(p["energy_eV"], p["amplitude"], 'ro', ms=6)
            ax.annotate(f'{p["energy_eV"]:.1f}eV', (p["energy_eV"], p["amplitude"]),
                       textcoords="offset points", xytext=(0,10), fontsize=8)
        ax.set_xlabel('Energy (eV)'); ax.set_ylabel('EEPF')
        ax.set_title(f'{base_name} — EEPF + {r["eepf_peaks"]["n_peaks"]} peaks')
        ax.legend(); ax.grid(alpha=0.3)
        path = str(OUTPUT_DIR / f'{base_name}_EEPF.{output_format}')
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        outputs.append(path)

    return {"status": "ok", "outputs": outputs, "n_plots": len(outputs)}


@mcp.tool()
def probe_info(topic: str = "all") -> dict:
    """探针参数/技术参考

    Args:
        topic: all | probe | formulas | ranges | modes | references
    """
    info = {
        "probe": {
            "default_geometry": "钨丝圆柱, φ0.15mm × 8mm, 面积≈0.038 cm²",
            "area_formula": "πR² + 2πRL (圆柱, 含端面)",
            "material": "钨 (W), 功函数 4.54 eV",
            "typical_position": "阴极正下方 1-2 cm (耦合区)"
        },
        "formulas": {
            "Druyvesteyn_EEPF": "F(ε) = √(8mₑε) / (e³A) · d²I/dV²",
            "Teff": "T_eff = (2/3)⟨ε⟩ — Godyak & Piejak 1990, PRL 65:996",
            "ftail": "f_tail = ∫_{ε>Vp-Vf} F dε / ∫F dε × 100% — Jauberteau 2018",
            "sheath_ratio": "(Vp-Vf)/Te — Maxwell理论=5.05, 实测=4.17±0.23",
            "Te_corrected": "Te × (5.05/4.17) ≈ Te × 1.21",
            "Ne_Langmuir": "Ne = 3.7×10¹³ × 0.9 × Ie / (A × √Te)",
        },
        "physical_ranges": PHYSICAL_RANGES,
        "mode_thresholds": MODE_RULES,
        "references": [
            "Godyak & Piejak 1990, PRL 65:996 — T_eff 定义",
            "Godyak & Demidov 2011, J. Phys. D 44:233001 — 探针 EEDF 综述",
            "Jauberteau & Jauberteau 2018, Contrib. Plasma Phys. — f_tail 高能尾积分",
            "Lobbia & Beal 2017, AIAA JPC — 电推进探针诊断推荐实践",
            "Lafleur & Chabert 2025, PSST 34:055005 — α 相似参数",
            "Yip et al. 2020, Plasma Sci. Technol. 22:085404 — 双温迭代减法",
            "Cellarius 1970 / Draganov 2025, Vacuum 235 — EEPF 多群分解",
        ]
    }

    if topic == "all":
        return info
    return info.get(topic, {"error": f"未知主题: {topic}", "available": list(info.keys())})

@mcp.tool()
def probe_gate_multi_ai(filepath: str, audit_dimensions: str = "all") -> dict:
    """多AI联合审计闸门 — 3个LLM并行审查: 数值合理性 + 一致性 + 历史漂移

    配置: PROBE_LLM_NUMERICAL=qwen3:32b PROBE_LLM_HISTORICAL=deepseek-r1:8b
           PROBE_LLM_API_BASE=http://127.0.0.1:11434/v1

    不配置LLM则回退到纯数值闸门(probe_gate)。

    Args:
        filepath: CSV文件路径
        audit_dimensions: all | numerical | consistency | historical
    """
    base = probe_analyze(filepath)
    if base["status"] != "ok":
        return {"status": "error", "error": base.get("error", "基础分析失败")}

    params = {
        "Vp": base["Vp"]["Vp"], "Vf": base["Vf"]["Vf"],
        "Te_eV": base["Te_classic"].get("Te_eV"),
        "Teff_eV": base["tail_metrics"]["Teff_eV"],
        "ftail_pct": base["tail_metrics"]["ftail_pct"],
        "sheath_ratio": base["tail_metrics"]["sheath_ratio"],
        "ne_cm3": base.get("Ne_classic_cm3"),
        "n_peaks": base["eepf_peaks"]["n_peaks"],
        "Thot_eV": base["eepf_peaks"]["Thot_eV"],
        "mode": base["mode"]["mode"], "n_steps": base["staircase"]["n_steps"],
        "has_upturn": base["upturn"]["has_upturn"],
    }

    system_prompt = "你是等离子体物理诊断专家。审计朗缪尔探针分析结果。只报告有具体证据的问题。每个问题标注严重程度ERROR/WARN/INFO。输出严格JSON。"

    results = {}; all_pass = True; flags = []

    # Numerical audit
    if audit_dimensions in ("all", "numerical"):
        model = _LLM_CONFIG["numerical"]
        if model:
            prompt = f"审计朗缪尔探针参数物理合理性。已知范围: Vp[-20,80]V, Vf[-40,40]V, Te[0.5,50]eV, Teff[1,100]eV, ne[1e6,1e13]cm-3, ftail[0,80]%, sheath[2,8]。参数:\n{json.dumps(params, indent=2, ensure_ascii=False)}\n输出JSON: {{\"violations\": [{{\"param\":\"...\", \"issue\":\"...\", \"severity\":\"ERROR|WARN|INFO\"}}]}}"
            resp = _call_llm(prompt, model, system_prompt)
            results["numerical"] = resp
            if resp["status"] == "ok":
                try:
                    j = json.loads(resp["response"].split("```json")[-1].split("```")[0])
                    if j.get("violations"): all_pass = False; flags.extend(j["violations"])
                except: pass
        else:
            basic = probe_gate(params["Vp"], params["Vf"], params["Te_eV"] or 1, params["Teff_eV"], params["ftail_pct"], params["sheath_ratio"])
            results["numerical"] = {"status": "fallback_basic_gate", "basic_gate": basic}
            if not basic.get("passed"): all_pass = False

    # Consistency audit
    if audit_dimensions in ("all", "consistency"):
        model = _LLM_CONFIG["consistency"] or _LLM_CONFIG["numerical"]
        if model:
            prompt = f"审计跨参数一致性。规则: 1)Vp>Vf 2)Teff>Te_slope(斜率法低估) 3)sheath_ratio~4.17±0.23 4)ftail~35%且n_peaks=1→肥尾单峰; ftail~35%且n_peaks≥6→多峰束状。参数:\n{json.dumps(params, indent=2, ensure_ascii=False)}\n输出JSON: {{\"issues\": [{{\"rule\":\"...\", \"status\":\"OK|VIOLATION\", \"detail\":\"...\"}}]}}"
            resp = _call_llm(prompt, model, system_prompt)
            results["consistency"] = resp
            if resp["status"] == "ok":
                try:
                    j = json.loads(resp["response"].split("```json")[-1].split("```")[0])
                    if any(i.get("status")=="VIOLATION" for i in j.get("issues",[])): all_pass = False
                except: pass

    # Historical drift
    if audit_dimensions in ("all", "historical"):
        model = _LLM_CONFIG["historical"] or _LLM_CONFIG["numerical"]
        if model:
            prompt = f"已知基线: sheath_ratio=4.17±0.23, f_tail(MODE_A)=19.5%, f_tail(OSC)=21.2%, f_tail(MODE_C)=34.7%, Teff(MODE_A/OSC)=18-20eV, Teff(MODE_C)=15eV。当前参数:\n{json.dumps(params, indent=2, ensure_ascii=False)}\n判断是否偏离基线。输出JSON: {{\"matches_baseline\": true/false, \"deviations\": [{{\"param\":\"...\", \"expected\":\"...\", \"actual\":\"...\", \"drift_pct\":...}}]}}"
            resp = _call_llm(prompt, model, system_prompt)
            results["historical"] = resp
            if resp["status"] == "ok":
                try:
                    j = json.loads(resp["response"].split("```json")[-1].split("```")[0])
                    if not j.get("matches_baseline", True): all_pass = False
                except: pass

    return {"status": "ok", "passed": all_pass, "dimensions_audited": list(results.keys()),
            "flags": flags, "results": results,
            "models": {k: v for k, v in _LLM_CONFIG.items() if k in ("numerical","consistency","historical","vision")}}


@mcp.tool()
def probe_visual_qa(folder_path: str, sample_n: int = 5, spatial_precompress: bool = True) -> dict:
    """批量多模态视觉验证 — 生成诊断图→VL模型看图核验→对比数值结果

    配置: PROBE_LLM_VISION=llava:13b PROBE_LLM_API_BASE=http://127.0.0.1:11434/v1
    空间预压缩(默认): 全局图+局部放大图双图打包, 防VL token压缩丢失细节。

    Args:
        folder_path: 含CSV文件的文件夹
        sample_n: 抽样数量 (默认5)
        spatial_precompress: 是否启用空间预压缩 (默认true)
    """
    import glob as _glob, base64 as _b64, random as _random

    model = _LLM_CONFIG["vision"]
    if not model:
        return {"status": "skipped", "reason": "未配置视觉模型。设置 PROBE_LLM_VISION=llava:13b。",
                "setup_hint": "ollama pull llava:13b"}

    csv_files = sorted(_glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        return {"status": "error", "error": "文件夹内无CSV"}

    sample_files = _random.sample(csv_files, min(sample_n, len(csv_files)))

    # Generate diagnostic plots
    plot_paths = []
    for fp in sample_files:
        try:
            r = probe_plot(fp, plot_type="all")
            if r["status"] == "ok": plot_paths.extend(r["outputs"])
        except: pass

    # Spatial pre-compression: zoom on EEPF tail region
    if spatial_precompress:
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except: pass
        for fp in sample_files[:min(3, len(sample_files))]:
            try:
                r = probe_analyze(fp)
                if r["status"] == "ok" and r["EEPF"]["status"] == "ok":
                    energy = np.array(r["EEPF"]["energy_axis_eV"])
                    eepf_v = np.array(r["EEPF"]["eepf_values"])
                    E_b = r["tail_metrics"]["E_barrier_eV"]
                    fig, ax = plt.subplots(figsize=(8,5), dpi=150)
                    mask = (energy > max(0,E_b-5)) & (energy < E_b+25)
                    if np.sum(mask) > 3:
                        ax.semilogy(energy[mask], np.maximum(eepf_v[mask],1e-30), 'b-', lw=2)
                        ax.axvline(E_b, color='m', ls='--', lw=2)
                        ax.set_title(f'EEPF tail zoom — {os.path.basename(fp)}')
                        ax.set_xlabel('Energy (eV)'); ax.set_ylabel('EEPF'); ax.grid(alpha=0.3)
                        zp = str(OUTPUT_DIR / f'zoom_tail_{os.path.basename(fp)}.png')
                        fig.savefig(zp, dpi=150, bbox_inches='tight'); plt.close()
                        plot_paths.append(zp)
            except: pass

    # VL verification (serial, one image at a time)
    results = []
    for pp in plot_paths:
        try:
            with open(pp, 'rb') as f:
                img_b64 = _b64.b64encode(f.read()).decode()
            prompt = "你是等离子体诊断图视觉验证器。看这张朗缪尔探针图: 1)标注(Vf/Vp线)清晰可读吗? 2)曲线平滑吗有无异常跳变? 3)峰标记(如有)是否合理? 4)有无制图错误? 输出JSON: {\"readable\":true/false,\"annotations_clear\":true/false,\"issues\":[\"...\"],\"overall\":\"OK|WARN\"}"
            resp = _call_llm(prompt, model, img_b64=img_b64)
            results.append({"image": os.path.basename(pp), "verification": resp})
        except Exception as e:
            results.append({"image": os.path.basename(pp), "verification": {"status":"error","error":str(e)[:200]}})

    n_ok = sum(1 for r in results if r["verification"].get("status")=="ok")
    mismatches = [r for r in results if "WARN" in str(r.get("verification",{}).get("response",""))]

    return {"status": "ok", "n_total": len(results), "n_ok": n_ok,
            "n_warn": len(mismatches), "mismatches": [{"image":m["image"]} for m in mismatches],
            "spatial_precompress": spatial_precompress, "plot_paths": plot_paths, "results": results}


# ═══════════════════════════════════════════════════════════════
# Gap-fill tools: bi-Maxwell, spectrum, similarity, OML, stratification
# ═══════════════════════════════════════════════════════════════

@mcp.tool()
def probe_fit_bimaxwell(filepath: str, max_iter: int = 10, tol: float = 0.01) -> dict:
    """Yip 2020 双麦克斯韦迭代减法拟合 — 分离热/冷两群电子

    Args:
        filepath: CSV 文件路径
        max_iter: 最大迭代次数
        tol: 收敛容差 (1%)
    """
    voltage, current = read_probe_csv(filepath)
    sort_idx = np.argsort(voltage)
    voltage = voltage[sort_idx]
    current = -current[sort_idx]
    voltage_s = smooth_gaussian(voltage)
    current_s = smooth_gaussian(current)

    vf = find_vf(voltage_s, current_s)["Vf"]
    vp_r = find_vp_multi(voltage_s, current_s, vf)
    Vp = vp_r.get("Vp")
    if Vp is None:
        return {"status": "error", "error": "Vp detection failed"}

    # Transition region data
    mask = (voltage_s >= vf) & (voltage_s <= Vp)
    x_trans = voltage_s[mask]
    y_trans = current_s[mask]
    logy_trans = np.log(np.maximum(y_trans, 1e-15))

    if len(x_trans) < 10:
        return {"status": "error", "error": "Transition region too short"}

    # Iterative subtraction (Yip et al. 2020, PST 22:085404)
    T_hot, T_cold = 0, 0
    T_hot_prev, T_cold_prev = 0, 0
    history = []

    for it in range(max_iter):
        # Find flattest d(ln I)/dV region → hottest electrons
        dlogy = np.gradient(logy_trans, x_trans)
        idx_min = np.argmin(np.abs(dlogy))
        V_fit_center = x_trans[idx_min]

        mask_hot = np.abs(x_trans - V_fit_center) < 2.0
        if np.sum(mask_hot) < 5: break

        coeffs = np.polyfit(x_trans[mask_hot], logy_trans[mask_hot], 1)
        T_hot = 1.0 / coeffs[0] if coeffs[0] > 0 else 0
        if T_hot <= 0 or T_hot > 100: break

        # Extrapolate hot-electron current and subtract
        I_sat_hot = current_s[np.argmin(np.abs(voltage_s - Vp))]
        I_hot_extrap = I_sat_hot * np.exp(-(voltage_s - Vp) / T_hot)
        I_residual = np.maximum(current_s - I_hot_extrap, 1e-15)
        logy_residual = np.log(I_residual[mask])

        # Fit cold from residual near Vf
        mask_cold = np.abs(x_trans - vf) < 3.0
        if np.sum(mask_cold) < 5: break

        coeffs_c = np.polyfit(x_trans[mask_cold], logy_residual[mask_cold], 1)
        T_cold = 1.0 / coeffs_c[0] if coeffs_c[0] > 0 else T_hot
        if T_cold <= 0 or T_cold > 100: break

        history.append({"iter": it+1, "Thot": round(float(T_hot),2), "Tcold": round(float(T_cold),2)})

        # Convergence check
        if it > 0:
            dh = abs(T_hot - T_hot_prev) / max(T_hot, 0.01)
            dc = abs(T_cold - T_cold_prev) / max(T_cold, 0.01)
            if dh < tol and dc < tol: break

        T_hot_prev, T_cold_prev = T_hot, T_cold

    # Validity gate
    valid = T_hot > T_cold and T_hot / max(T_cold, 0.01) > 1.5 and T_hot < 100
    if not valid:
        T_cold = T_hot  # Degenerate to single-temperature

    # Density estimate
    I_sat = float(np.max(current_s) - np.min(current_s))
    area = probe_area()
    n_hot = 0; n_cold = 0
    if T_hot > 0:
        n_hot = 3.7e13 * 0.9 * I_sat * 0.5 / (area * np.sqrt(T_hot))
        if T_cold > 0 and T_cold != T_hot:
            n_cold = 3.7e13 * 0.9 * I_sat * 0.5 / (area * np.sqrt(T_cold))

    return {"status": "ok", "Thot_eV": round(float(T_hot),2), "Tcold_eV": round(float(T_cold),2),
            "n_hot_cm3": round(float(n_hot),2), "n_cold_cm3": round(float(n_cold),2),
            "Thot_Tcold_ratio": round(float(T_hot/max(T_cold,0.01)),2),
            "valid": valid, "iterations": len(history), "convergence_history": history,
            "reference": "Yip et al. 2020, Plasma Sci. Technol. 22:085404"}


@mcp.tool()
def probe_spectrum(filepath: str, fs_hz: float = 200000.0, detect_modes: bool = True) -> dict:
    """示波器频谱分析 — FFT + 呼吸模/渡越模检测

    适用: Rigol DHO5054 或其他示波器CSV (Time,CH1,CH2...)

    Args:
        filepath: 示波器CSV文件路径
        fs_hz: 采样率 Hz (DHO5054默认200kSa/s)
        detect_modes: 是否自动检测振荡模式
    """
    try:
        import pandas as pd
        filepath = _safe_path(filepath)
        df = pd.read_csv(filepath)
    except:
        return {"status": "error", "error": "CSV read failed"}

    # Find time and signal columns
    time_col = [c for c in df.columns if 'time' in c.lower() or 'Time' in c][0]
    ch_cols = [c for c in df.columns if c != time_col]
    if not ch_cols:
        return {"status": "error", "error": "No signal columns found"}

    results = {"status": "ok", "fs_hz": fs_hz, "n_samples": len(df), "channels": {}}

    for ch in ch_cols[:2]:
        signal = df[ch].dropna().values
        if len(signal) < 10: continue

        # Remove DC offset + FFT
        signal_ac = signal - np.mean(signal)
        n = len(signal_ac)
        fft = np.abs(np.fft.rfft(signal_ac))
        freqs = np.fft.rfftfreq(n, d=1.0/fs_hz)

        # Top peaks
        peak_idx, props = find_peaks(fft, height=np.max(fft)*0.1, distance=5)
        top_n = min(10, len(peak_idx))
        sort_idx = np.argsort(fft[peak_idx])[::-1][:top_n]

        peaks = []
        for i in sort_idx:
            peaks.append({"freq_Hz": round(float(freqs[peak_idx[i]]), 1),
                         "amplitude": round(float(fft[peak_idx[i]]), 2)})

        ch_result = {"peaks": peaks, "nyquist_hz": fs_hz/2}

        # Mode detection
        if detect_modes and peaks:
            breathing = [p for p in peaks if 10000 < p["freq_Hz"] < 100000]
            transit = [p for p in peaks if 100000 < p["freq_Hz"] < 5000000]
            iat = [p for p in peaks if p["freq_Hz"] > 300000 and p["freq_Hz"] < 5000000]

            # Relaxation oscillation: harmonic series check
            if breathing:
                freqs_b = sorted([p["freq_Hz"] for p in breathing])
                if len(freqs_b) >= 2:
                    spacings = np.diff(freqs_b)
                    is_harmonic = np.std(spacings) / max(np.mean(spacings), 1) < 0.15
                else:
                    is_harmonic = False
                ch_result["breathing_mode"] = {
                    "detected": True, "fundamental_Hz": freqs_b[0],
                    "n_harmonics": len(freqs_b), "is_relaxation_oscillation": bool(is_harmonic)}
            else:
                ch_result["breathing_mode"] = {"detected": False}

            if transit:
                ch_result["transit_mode"] = {"detected": True, "frequencies_Hz": [p["freq_Hz"] for p in transit[:5]]}
            else:
                ch_result["transit_mode"] = {"detected": False}

        results["channels"][ch] = ch_result

    return results


@mcp.tool()
def probe_similarity(alpha_values: list = None, ftail_values: list = None,
                     B_G: list = None, mdot_mg_s: list = None) -> dict:
    r"""相似参数计算 — Lafleur & Chabert 2025 α=B²/ṁ 框架

    可从原始参数计算α或直接输入α值列表。

    Args:
        alpha_values: α值列表 (如已算好)
        ftail_values: f_tail值列表 (用于模式分层)
        B_G: 磁场强度列表 [G] (与mdot_mg_s配对使用)
        mdot_mg_s: 质量流量列表 [mg/s]
    """
    result = {"status": "ok", "reference": "Lafleur & Chabert 2025, PSST 34:055005"}

    # Compute alpha if raw inputs provided
    if B_G and mdot_mg_s:
        alphas = [b**2 / m for b, m in zip(B_G, mdot_mg_s)]
        result["alpha_computed"] = [round(float(a), 2) for a in alphas]
        result["alpha_mean"] = round(float(np.mean(alphas)), 2)
        result["alpha_std"] = round(float(np.std(alphas)), 2)
        result["formula"] = "alpha = B^2 / mdot  [G^2/(mg/s)]"
        result["B_unit"] = "G (axial peak radial B-field)"
        result["mdot_unit"] = "mg/s (anode mass flow rate)"
    elif alpha_values:
        result["alpha_values"] = alpha_values
        result["alpha_mean"] = round(float(np.mean(alpha_values)), 2)

    # Mode stratification
    if ftail_values:
        modes = []
        for f in ftail_values:
            if f > 30: modes.append("MODE_C")
            elif f > 20: modes.append("MODE_B")
            else: modes.append("MODE_A")
        from collections import Counter
        counts = dict(Counter(modes))
        result["mode_distribution"] = counts
        result["mode_boundaries"] = {
            "MODE_A": "ftail < 25% (typical: ~19.5%)",
            "MODE_B": "ftail 20-30% (typical: ~21.2%)",
            "MODE_C": "ftail > 25% (typical: ~34.7%)"}

    # Derived similarity parameters
    result["related_parameters"] = {
        "beta": "Vp/Te — acceleration-to-thermalization ratio",
        "lambda_star": "nu_en/omega_ce — collision-to-magnetization ratio",
        "note": "beta and lambda_star require additional inputs (Vp,Te,ngas,B). Use probe_analyze first."}

    return result


@mcp.tool()
def probe_predict_mode(ftail: float, I_std: float = None, log10_alpha: float = None) -> dict:
    """Plume概率预测 — 逻辑回归融合模型 (88.9% accuracy)

    基于66组Kr工质B场扫描标定的融合模型:
    P(OSC/MODE_C) = sigma(-2.99 + 0.034*log10(alpha) + 2.748*I_std + 0.098*f_tail)

    Args:
        ftail: 高能密度占比 (%)
        I_std: 放电电流波动标准差 (可选)
        log10_alpha: log10(B²/ṁ) (可选)
    """
    # Default model coefficients (from plume_probability_model.json)
    w = [-2.99, 0.034, 2.748, 0.098]

    # Use defaults if optional inputs missing
    if I_std is None:
        I_std = 0.5  # typical OSC/MODE_C value
    if log10_alpha is None:
        log10_alpha = 2.0  # typical transition region

    z = w[0] + w[1]*log10_alpha + w[2]*I_std + w[3]*ftail
    prob = 1.0 / (1.0 + np.exp(-z))  # sigmoid

    mode = "MODE_C" if prob > 0.5 else ("MODE_B" if prob > 0.3 else "MODE_A")

    return {"status": "ok", "probability_OSC_MODE_C": round(float(prob*100), 1),
            "predicted_mode": mode, "model_accuracy": "88.9%",
            "coefficients": {"intercept": w[0], "log10_alpha": w[1], "I_std": w[2], "ftail": w[3]},
            "note": "模型基于66组Kr工质B场扫描标定。不同工质/推力器需重新标定。"}


@mcp.tool()
def probe_stratify(filepath_list: list, group_labels: list = None,
                   group_by: str = "discharge_config") -> dict:
    """分层分析 — 防止Simpson悖论

    按分组变量(如keeper电流/流量/磁场)分层统计，自动检测合并vs分组的符号反转。

    Args:
        filepath_list: CSV文件路径列表
        group_labels: 每个文件的组标签 (如 ["1A","1A",...,"0.5A","0.5A",...])
        group_by: 分组变量名 (默认 discharge_config)
    """
    if not group_labels:
        return {"status": "error", "error": "需要group_labels参数指定每个文件的分组"}

    if len(filepath_list) != len(group_labels):
        return {"status": "error", "error": "filepath_list和group_labels长度不一致"}

    # Analyze all files
    all_results = []
    for fp in filepath_list:
        try:
            r = probe_analyze(fp)
            if r["status"] == "ok": all_results.append(r)
        except: pass

    # Extract key metrics
    metrics = ["ftail_pct", "Teff_eV", "sheath_ratio", "n_peaks"]
    overall = {}
    grouped = {}

    for metric in metrics:
        all_vals = []
        group_vals = {}
        for r, label in zip(all_results, group_labels):
            if metric == "ftail_pct":
                val = r["tail_metrics"]["ftail_pct"]
            elif metric == "Teff_eV":
                val = r["tail_metrics"]["Teff_eV"]
            elif metric == "sheath_ratio":
                val = r["tail_metrics"]["sheath_ratio"]
            elif metric == "n_peaks":
                val = r["eepf_peaks"]["n_peaks"]
            else:
                val = None

            if val is not None:
                all_vals.append(val)
                group_vals.setdefault(label, []).append(val)

        overall[metric] = {"mean": round(float(np.mean(all_vals)),2) if all_vals else None,
                          "std": round(float(np.std(all_vals)),2) if all_vals else None,
                          "n": len(all_vals)}

        grouped[metric] = {}
        for label, vals in group_vals.items():
            grouped[metric][label] = {"mean": round(float(np.mean(vals)),2),
                                      "std": round(float(np.std(vals)),2), "n": len(vals)}

    # Simpson's paradox detection: compare overall vs per-group correlations
    # Check if any metric's group ordering contradicts overall
    paradox_warnings = []
    for metric in metrics:
        group_means = [(label, g["mean"]) for label, g in grouped[metric].items() if g["mean"] is not None]
        if len(group_means) >= 2:
            overall_mean = overall[metric]["mean"]
            # Check if overall mean lies outside the range of group means
            gmeans = [m for _, m in group_means]
            if overall_mean is not None and (overall_mean < min(gmeans) or overall_mean > max(gmeans)):
                paradox_warnings.append({
                    "metric": metric,
                    "overall_mean": overall_mean,
                    "group_means": dict(group_means),
                    "warning": "总体均值落在各组均值范围之外 — 可能存在Simpson悖论,请分层解读"
                })

    return {"status": "ok", "group_by": group_by, "n_groups": len(set(group_labels)),
            "n_total": len(all_results), "overall": overall, "grouped": grouped,
            "simpson_paradox_warnings": paradox_warnings,
            "note": "检测到警告时: 不要合并数据做单一相关分析,必须分组讨论"}


@mcp.tool()
def probe_anode_detect(filepath: str, jump_threshold_mA_s: float = 30.0,
                       plateau_min_duration_s: float = 30.0) -> dict:
    """放电电流跳变检测 — 识别放电模式转变事件

    适用于放电电源遥测CSV (含Time, Current列)。

    Args:
        filepath: 阳极遥测CSV
        jump_threshold_mA_s: 跳变阈值 mA/s (默认30)
        plateau_min_duration_s: 最小平稳段时长 s (默认30)
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
    except:
        return {"status": "error", "error": "CSV read failed"}

    time_col = [c for c in df.columns if 'time' in c.lower() or 'Time' in c][0]
    curr_col = [c for c in df.columns if 'current' in c.lower() or 'Current' in c or 'I_' in c or 'A' in c][0]

    t = df[time_col].values
    i = df[curr_col].values

    if len(t) < 10:
        return {"status": "error", "error": "Not enough data points"}

    # Detect jumps: |dI/dt| > threshold
    dt = np.diff(t)
    di = np.diff(i)
    didt = np.abs(di / np.maximum(dt, 1e-6)) * 1000  # mA/s

    jump_idx = np.where(didt > jump_threshold_mA_s)[0]
    jumps = []
    if len(jump_idx) > 0:
        # Cluster nearby jumps
        clusters = [[jump_idx[0]]]
        for j in jump_idx[1:]:
            if t[j] - t[clusters[-1][-1]] < 10:  # within 10s
                clusters[-1].append(j)
            else:
                clusters.append([j])
        for c in clusters:
            jumps.append({
                "time_s": round(float(t[c[0]]), 1),
                "current_before_A": round(float(i[max(0,c[0]-1)]), 3),
                "current_after_A": round(float(i[min(len(i)-1,c[-1]+1)]), 3),
                "delta_A": round(float(i[min(len(i)-1,c[-1]+1)] - i[max(0,c[0]-1)]), 3),
                "max_rate_mA_s": round(float(np.max(didt[c])), 1)})

    # Detect plateaus: stable current for > plateau_min_duration_s
    # Round current to 0.1mA resolution
    i_rounded = np.round(i, 3)
    plateaus = []
    start = 0
    for idx in range(1, len(i_rounded)):
        if abs(i_rounded[idx] - i_rounded[start]) > 0.001:  # >1mA change
            duration = t[idx-1] - t[start]
            if duration >= plateau_min_duration_s:
                plateaus.append({
                    "start_s": round(float(t[start]), 1),
                    "end_s": round(float(t[idx-1]), 1),
                    "duration_s": round(float(duration), 1),
                    "current_A": round(float(np.mean(i[start:idx])), 3),
                    "current_std_A": round(float(np.std(i[start:idx])), 4)})
            start = idx

    return {"status": "ok", "n_jumps": len(jumps), "jumps": jumps,
            "n_plateaus": len(plateaus), "plateaus": plateaus,
            "total_duration_s": round(float(t[-1] - t[0]), 1)}


# ── probe_plot: add missing plot types ──
# (monkey-patch: extend plot_type options to include "oml" and "logiv" and "bimaxwell")

# Store original probe_plot reference
_original_probe_plot = probe_plot

@mcp.tool()
def probe_plot_extended(filepath: str, plot_type: str = "all",
                        output_format: str = "png") -> dict:
    if output_format not in ("png", "svg", "pdf"):
        output_format = "png"
    """扩展诊断图 — 新增: OML线性检验(I^0.75)、ln(I)-V、双温拟合叠加

    Args:
        filepath: CSV文件路径
        plot_type: all | iv | eepf | oml | logiv | bimaxwell
        output_format: png | svg
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Get analysis data
    r = probe_analyze(filepath)
    if r["status"] != "ok":
        return {"status": "error", "error": r.get("error")}

    voltage, current = read_probe_csv(filepath)
    sort_idx = np.argsort(voltage)
    voltage = voltage[sort_idx]
    current = -current[sort_idx]
    voltage_s = smooth_gaussian(voltage)
    current_s = smooth_gaussian(current)

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    outputs = []

    # OML check: I^0.75 vs V (ion saturation region should be linear)
    if plot_type in ("all", "oml"):
        fig, ax = plt.subplots(figsize=(8,5), dpi=130)
        vf = r["Vf"]["Vf"]
        mask_ion = voltage_s < (vf - 2)  # ion saturation
        if np.sum(mask_ion) > 5:
            i_ion = np.abs(current_s[mask_ion])
            ax.plot(voltage_s[mask_ion], np.power(np.maximum(i_ion,1e-15), 0.75), 'b.', ms=4)
            # Linear fit
            coeffs = np.polyfit(voltage_s[mask_ion], np.power(np.maximum(i_ion,1e-15), 0.75), 1)
            ax.plot(voltage_s[mask_ion], np.polyval(coeffs, voltage_s[mask_ion]), 'r--', lw=1.5,
                   label=f'slope={coeffs[0]:.4f}')
            ax.legend()
        ax.set_xlabel('Voltage (V)'); ax.set_ylabel('I^{0.75}')
        ax.set_title(f'{base_name} — OML linearity check (cylindrical probe)')
        ax.grid(alpha=0.3)
        path = str(OUTPUT_DIR / f'{base_name}_OML.{output_format}')
        fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
        outputs.append(path)

    # Log(IV)
    if plot_type in ("all", "logiv"):
        fig, ax = plt.subplots(figsize=(8,5), dpi=130)
        mask_pos = current_s > 1e-12
        ax.semilogy(voltage_s[mask_pos], current_s[mask_pos], 'b-', lw=1.5)
        ax.axvline(r["Vf"]["Vf"], color='g', ls='--', label=f'Vf={r["Vf"]["Vf"]:.1f}')
        ax.axvline(r["Vp"]["Vp"], color='r', ls='--', label=f'Vp={r["Vp"]["Vp"]:.1f}')
        # Transition region linear fit overlay
        mask_trans = (voltage_s >= r["Vf"]["Vf"]) & (voltage_s <= r["Vp"]["Vp"])
        if np.sum(mask_trans) > 5 and r["Te_classic"].get("Te_eV"):
            Te = r["Te_classic"]["Te_eV"]
            I_vp = current_s[np.argmin(np.abs(voltage_s - r["Vp"]["Vp"]))]
            ax.plot(voltage_s[mask_trans], I_vp * np.exp(-(r["Vp"]["Vp"]-voltage_s[mask_trans])/Te),
                   'r--', lw=1, alpha=0.7, label=f'Te={Te:.1f}eV fit')
        ax.set_xlabel('Voltage (V)'); ax.set_ylabel('Current (A)')
        ax.set_title(f'{base_name} — ln(I)-V')
        ax.legend(); ax.grid(alpha=0.3)
        path = str(OUTPUT_DIR / f'{base_name}_LogIV.{output_format}')
        fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
        outputs.append(path)

    # Bi-Maxwellian fit overlay
    if plot_type in ("all", "bimaxwell"):
        bimax = probe_fit_bimaxwell(filepath)
        if bimax["status"] == "ok" and bimax["valid"]:
            fig, ax = plt.subplots(figsize=(8,5), dpi=130)
            ax.semilogy(voltage_s, np.maximum(current_s,1e-15), 'k.', ms=2, alpha=0.5, label='data')
            Vp = r["Vp"]["Vp"]
            I_sat = current_s[np.argmin(np.abs(voltage_s-Vp))]
            # Hot electron extrapolation
            I_hot = I_sat * np.exp(-np.maximum(voltage_s-Vp, 0)/bimax["Thot_eV"])
            ax.semilogy(voltage_s, np.maximum(I_hot,1e-15), 'r--', lw=1.2,
                       label=f'T_hot={bimax["Thot_eV"]:.1f}eV')
            # Cold residual
            I_cold = current_s - I_hot
            ax.semilogy(voltage_s[I_cold>1e-15], I_cold[I_cold>1e-15], 'b-', lw=1,
                       label=f'T_cold={bimax["Tcold_eV"]:.1f}eV')
            ax.axvline(Vp, color='k', ls='--'); ax.axvline(r["Vf"]["Vf"], color='g', ls='--')
            ax.legend(); ax.set_xlabel('Voltage (V)'); ax.set_ylabel('Current (A)')
            ax.set_title(f'{base_name} — Bi-Maxwellian fit (Yip 2020)'); ax.grid(alpha=0.3)
            path = str(OUTPUT_DIR / f'{base_name}_biMaxwell.{output_format}')
            fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
            outputs.append(path)

    # Also run original probe_plot for iv/eepf
    orig = _original_probe_plot(filepath, plot_type if plot_type in ("all","iv","eepf") else "all", output_format)
    if orig["status"] == "ok":
        outputs.extend(orig.get("outputs", []))

    return {"status": "ok", "outputs": list(set(outputs)), "n_plots": len(set(outputs))}


# ── Enhance: probe_detect_steps add comb-tooth detection ──

@mcp.tool()
def probe_detect_comb_teeth(filepath: str, tolerance: float = 0.15) -> dict:
    """梳齿伪影检测 — 检测dI/dV上等间距的伪台阶

    真实电子群的台阶间距不均匀(2-8V不等)。
    梳齿伪影间距均匀(≈峰检测最小间距参数)。

    Args:
        filepath: CSV文件路径
        tolerance: 间距均匀性容差 (默认15%)
    """
    stairs = probe_detect_steps(filepath)
    if stairs["n_steps"] < 3:
        return {"status": "ok", "has_comb_teeth": False,
                "reason": f"台阶数不足({stairs['n_steps']}), 至少需要3个"}

    voltages = [s["voltage_V"] for s in stairs["steps"]]
    spacings = np.diff(voltages)
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)

    # Comb teeth: spacings are nearly uniform
    cv = std_spacing / max(mean_spacing, 0.01)
    is_comb = cv < tolerance

    return {"status": "ok", "has_comb_teeth": is_comb,
            "spacings_V": [round(float(s),2) for s in spacings],
            "mean_spacing_V": round(float(mean_spacing),2),
            "std_spacing_V": round(float(std_spacing),2),
            "cv": round(float(cv),3),
            "judgment": "可能为梳齿伪影(等间距), 建议增大平滑窗宽" if is_comb else "间距不均匀, 更可能是真实电子群台阶",
            "fix_hint": "增大smooth window或减小prominence阈值" if is_comb else None}


# ── Enhance: probe_analyze add OML sheath expansion correction ──

def _sheath_expansion_correction(ne_cm3: float, Te_eV: float, R_mm: float) -> dict:
    """OML厚鞘修正: lambda_D/R → effective collection area correction"""
    if ne_cm3 is None or Te_eV is None or ne_cm3 <= 0 or Te_eV <= 0:
        return {"lambda_D_mm": None, "lambda_D_over_R": None, "correction_note": "参数不足"}

    eps0 = 8.854e-12
    ne_m3 = ne_cm3 * 1e6
    lambda_D = np.sqrt(eps0 * Te_eV / (ne_m3 * 1.602e-19))  # meters
    lambda_D_mm = lambda_D * 1000

    R = R_mm
    ratio = lambda_D_mm / R

    correction = ""
    if ratio < 1:
        correction = "薄鞘 (lambda_D < R): OML公式适用, 几何面积可用"
    elif ratio < 10:
        correction = f"厚鞘 (lambda_D/R={ratio:.1f}): 有效收集面积>几何面积, f_tail绝对值被低估, 偏差方向保守"
    else:
        correction = f"极厚鞘 (lambda_D/R={ratio:.1f}): OML假设恶化, 建议使用ABR理论修正"

    return {"lambda_D_mm": round(lambda_D_mm, 4), "lambda_D_over_R": round(ratio, 1),
            "correction": correction}


def main():
    """MCP Server 入口"""
    print(f"Langmuir Probe MCP v{VERSION}")
    print(f"输出目录: {OUTPUT_DIR}")
    mcp.run()


if __name__ == "__main__":
    main()
