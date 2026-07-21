# Langmuir Probe MCP v1.0 — 产品描述与技术白皮书

> **定位**: 面向 AI Agent 的朗缪尔探针 IV 曲线分析 MCP Server
> **交付日期**: 2026-07-21 | **版本**: v1.0.0 | **许可证**: MIT | **语言**: Python 3.10+
> **代码位置**: `交付成品_商业MCP/langmuir-probe-mcp/`

---

## 1. 一句话概述

**把一根金属丝的电压-电流扫描数据，自动翻译成等离子体状态参数和放电模式判别结果。**

AI Agent（Claude Code / Cursor / Windsurf 等）通过 MCP 协议调用本工具，无需写任何代码，直接从原始 CSV 得到 EEPF 电子能量分布、非麦克斯韦指标、模式分类和质量闸门报告。

---

## 2. 解决的痛点

| 痛点 | 本 MCP 的方案 |
|------|-------------|
| 朗缪尔探针 IV 曲线分析依赖 MATLAB，不能脱离商业软件 | 纯 Python（numpy/scipy），零 MATLAB 依赖 |
| 传统分析只给 Te/Ne 两个标量，丢失电子能量分布的形状信息 | 输出 17 个参数 + EEPF 全量数组 + 多峰分解 |
| 探针数据处理步骤繁琐：平滑→找 Vf→找 Vp→求导→积分→... | 一个 `probe_analyze` 调用走完 L0→L4 全链路 |
| 缺乏自动化质量闸门，异常数据混入后续分析 | 内置三层闸门：数值范围/跨参数一致性/历史漂移 |
| 批量处理 66 组实验数据需要手动循环 | `probe_batch` 一键批量，输出 result.txt |
| 非等离子体专业人士看不懂 EEPF | `probe_info` 内置公式参考+物理范围+文献池 |

---

## 3. 技术架构

```
┌─────────────────────────────────────────────────┐
│ AI Agent (Claude Code) │
│ "帮我分析这个探针数据,判断放电模式" │
└─────────────────┬───────────────────────────────┘
 │ MCP protocol (JSON-RPC over stdio)
┌─────────────────▼───────────────────────────────┐
│ Langmuir Probe MCP Server │
│ │
│  probe_analyze  ←── L0→L4 全链路 │
│  probe_batch ←── 批量 + result.txt │
│  probe_eepf ←── Druyvesteyn EEPF │
│  probe_detect_modes ←── SPOT/OSC/PLUME │
│  probe_detect_steps ←── dI/dV 台阶检测 │
│  probe_detect_upturn ←── 末端上翘检测 │
│  probe_gate ←── 质量闸门 │
│  probe_compare  ←── 两装置对比 │
│  probe_plot ←── 诊断图 (PNG/SVG) │
│  probe_info ←── 技术参考 │
│ │
│  核心引擎: │
│  ┌──────────────────────────────────────────┐ │
│  │ CSV→排序取反→高斯平滑→dI/dV/d²I/dV² │ │
│  │ → Vf(|I|最小)→Vp(dI/dV多峰) │ │
│  │ → Te(lnI斜率)→Ne(OML公式) │ │
│  │ → EEPF(Druyvesteyn: F∝√ε·d²I/dV²) │ │
│  │ → f_tail(高能占比)→T_eff(有效温度) │ │
│  │ → Te_corr(鞘层比修正)→sheath_ratio │ │
│  │ → EEPF多峰分解(sgolay+峰检测) │ │
│  │ → 台阶检测(四项伪影排除)→上翘检测 │ │
│  │ → 模式判别(SPOT/OSC/PLUME)→闸门 │ │
│  └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 4. 10 个 MCP 工具详细说明

### 4.1 `probe_analyze` — 单条 IV 全量分析 (主力工具)

**输入**: CSV 文件路径 + 探针几何(可选)
**输出**: 完整 JSON — Vf, Vp, Te, Ne, EEPF, f_tail, T_eff, Te_corr, sheath_ratio, EEPF 峰位, 台阶数, 上翘深度, 模式分类, 闸门结果

**覆盖的处理步骤**:
1. CSV 解析 (自动跳过 IT2801 表头, 适配多种编码)
2. 电压排序 + 电流取反 (电子流为正)
3. 高斯平滑 (窗宽 9 × 5 次迭代, `scipy.ndimage.gaussian_filter1d`)
4. Vf 检测: |I| 最小点 (取第一个, v2 修复)
5. Vp 检测: dI/dV 多峰 → 最高电压峰 = 主 Vp (去重 <0.5V)
6. 经典 Te: 过渡区 ln(I) 斜率法
7. 经典 Ne: OML 公式 `3.7×10¹³ × Ie / (A×√Te)`
8. EEPF 重建: Druyvesteyn `F(ε) ∝ √(8mₑε)/e³/A · d²I/dV²`
9. f_tail: ∫_{ε>Vp-Vf} F dε / ∫F dε × 100% (密度口径)
10. T_eff: (2/3)⟨ε⟩ (Godyak & Piejak 1990)
11. Te_corr: Te × (5.05/4.17) (鞘层比修正)
12. sheath_ratio: (Vp-Vf)/Te (本征参数)
13. EEPF 多峰分解: 三段 sgolay + 峰检测 (Cellarius 1970 + Draganov 2025)
14. dI/dV 台阶检测 + 四项伪影排除
15. dI/dV 末端上翘检测
16. 模式判别: SPOT / OSCILLATING / PLUME
17. 三层质量闸门

**调用示例**:
```
AI: "分析 E:/data/sweep_00001.csv, 探针直径 0.15mm 长 8mm"

→ probe_analyze(filepath="E:/data/sweep_00001.csv", R_mm=0.075, L_mm=8.0)
→ {
 "Vp": {"Vp": 35.8, "n_peaks": 3},
 "Te_classic": {"Te_eV": 8.1},
 "tail_metrics": {"ftail_pct": 34.7, "Teff_eV": 18.3, "sheath_ratio": 4.17},
 "mode": {"mode": "PLUME", "confidence": "high"},
 "gate": {"passed": true}
  }
```

---

### 4.2 `probe_batch` — 批量处理

**输入**: 文件夹路径
**输出**: 汇总 JSON + 自动生成 `result.txt` (制表符分隔, 15 列)

自动遍历文件夹内所有 CSV, 逐文件调用 `probe_analyze`, 汇总统计。

---

### 4.3 `probe_eepf` — 单独 EEPF 重建

**输入**: CSV 路径 + Vp(可选)
**输出**: 能量轴数组 + EEPF 值数组 + ne_EEPF + Teff

用于只需要 EEPF 不需要全量分析的场景。

---

### 4.4 `probe_detect_modes` — 模式判别

**输入**: f_tail 值列表 + sheath_ratio 列表(可选)
**输出**: 每个点的模式标签 + 统计 + 主导模式

**判别逻辑**:
| 模式 | 条件 | 物理特征 |
|------|------|---------|
| SPOT | f_tail < 25% | 恒压 300V, 纹波 <1% |
| OSCILLATING | 20% < f_tail < 30% | 呼吸模 ~30kHz, 纹波 5-20% |
| PLUME | f_tail > 25% | 恒流转压, 纹波 >20% |

---

### 4.5 `probe_detect_steps` — dI/dV 台阶检测

**输入**: CSV 路径 + 突出度阈值
**输出**: 台阶数 + 每个台阶的电压/突出度/宽度 + 四项伪影排除测试结果

**四项排除**:
1. 仅出现在电子排斥区 (V > Vf)
2. 不锁绝对电压网格
3. 不聚 SMU 量程边界
4. 随 Vf 漂移

---

### 4.6 `probe_detect_upturn` — 末端上翘检测

**输入**: CSV 路径
**输出**: 是否有上翘 + 上翘深度(归一化) + 上翘电压范围

**算法**: dI/dV 末端 15 点偏离 Vp→+35V 线性外推的最大距离, 归一化至 dI/dV 峰值。

---

### 4.7 `probe_gate` — 质量闸门

**输入**: Vp, Vf, Te, Teff, ftail, sheath_ratio
**输出**: 通过/不通过 + 违规列表 + 一致性检查

**物理范围** (来自 实验标定):

| 参数 | 范围 | 单位 |
|------|------|------|
| Vp | -20 ~ 80 | V |
| Vf | -40 ~ 40 | V |
| Te | 0.5 ~ 50 | eV |
| Teff | 1 ~ 100 | eV |
| ne | 10⁶ ~ 10¹³ | cm⁻³ |
| ftail | 0 ~ 80 | % |
| sheath_ratio | 2 ~ 8 | — |

**一致性规则**: Vp > Vf, Teff > Te, 鞘层比稳定在 4.17±0.23

---

### 4.8 `probe_compare` — 两装置对比

**输入**: 两个 CSV 路径 + 标签
**输出**: Vp/Vf/Te/ftail/sheath_ratio/Teff/n_peaks 逐项对比 + "同参数不同质"检测

**特色**: 自动检测不同放电配置的"同参数不同质"陷阱— Teff 相同但 EEPF 形状 KS=1.00。

---

### 4.9 `probe_plot` — 诊断图生成

**输入**: CSV 路径 + 图表类型
**输出**: PNG/SVG 文件路径

**图表类型**:
- `iv` — I-V 曲线 + dI/dV 双轴图 (标注 Vf/Vp)
- `eepf` — EEPF 分布图 (标注鞘层势垒线 + f_tail 值 + 峰位)
- `all` — 两者都生成

---

### 4.10 `probe_info` — 技术参考

**输入**: 主题 (all/probe/formulas/ranges/modes/references)
**输出**: 探针参数/公式/物理范围/模式阈值/文献池

内置 7 篇核心文献的完整引用信息。

---

## 5. 技术指标

| 指标 | 数值 |
|------|------|
| 覆盖分析技术 | 38 项 (源自 MATLAB v4 51 项技术的纯 Python 实现) |
| 输出参数 | 17 列 (Vp, Ne, Vf, Te, Ie, ne_EEPF, Te_EEPF, Thot, nhot, Tcold, ncold, peak_count, ftail, Teff, Te_corr, sheath_ratio) |
| 输入格式 | IT2801 CSV (3 行表头 + V/I 两列), 自动适配编码 |
| 探针类型 | 圆柱单探针 (可配置半径/长度) |
| EEPF 重建 | Druyvesteyn 公式 (Godyak & Demidov 2011) |
| 模式判别 | SPOT / OSCILLATING / PLUME (基于 66 组 工质 B 场扫描标定) |
| 质量闸门 | 9 参数物理范围 + 3 条一致性规则 |
| 依赖 | numpy, scipy, matplotlib, mcp |
| 安装 | `pip install -e .` |
| 输出格式 | JSON (tools) / PNG|SVG (plots) / TSV (batch result.txt) |
| 安全 | 路径穿越防护 (`PROBE_ALLOWED_BASE` 环境变量) |
| 许可证 | MIT (开源) |

---

## 6. 与 MATLAB v4 的对应关系

本 MCP 是 MATLAB `ZYCdensity_source_disp_v4.m` 的纯 Python 移植和扩展:

| MATLAB v4 函数 | MCP 实现 | 说明 |
|---------------|---------|------|
| `Multi_Smooth` | `scipy.ndimage.gaussian_filter1d` | 高斯平滑 |
| `Find_Vf_v2` | `find_vf()` | \|I\| 最小点 |
| `Find_Vp_multi` | `find_vp_multi()` | dI/dV 多峰 + 0.5V 去重 |
| `Find_Te_single` | `find_te_classic()` | ln(I) 斜率法 |
| `Find_Ne` | `find_ne_classic()` | OML 公式 |
| `Find_EEPF` | `find_eepf()` | Druyvesteyn |
| `Find_tail_metrics` | `find_tail_metrics()` | f_tail + T_eff + Te_corr + sheath_ratio |
| `Find_Te_EEPF_multi` | `detect_eepf_peaks()` | 三段 sgolay + 峰检测 |
| — | `detect_staircase()` | 台阶检测 (MCP 新增) |
| — | `detect_upturn()` | 上翘检测 (MCP 新增) |
| — | `classify_mode()` | 模式判别 (MCP 新增) |
| — | `gate_check()` | 质量闸门 (MCP 新增) |

---

## 7. 适用场景

### 已验证: 低温等离子体诊断
- 多组磁场/流量扫描 (模式转变研究)
- 多组对照放电实验
- 等离子体源下游诊断区
- 电子密度范围: 10⁸ ~ 10¹⁰ cm⁻³
- 电子温度范围: 0.5 ~ 50 eV

### 可扩展: 
- 液滴/电喷雾等离子体源 (弱电离, 低密度 — 需调整平滑参数和电压范围)
- RF/ICP 等离子体源
- DC 辉光放电
- 任何使用圆柱单探针的等离子体诊断场景

---

## 8. 安装与使用

### 安装
```bash
cd 交付成品_商业MCP/langmuir-probe-mcp
pip install -e .
```

### MCP 配置 (Claude Code / Cursor)
```json
{
  "mcpServers": {
 "langmuir-probe": {
 "command": "python",
 "args": ["-m", "langmuir_mcp.server"],
 "env": {
 "PROBE_ALLOWED_BASE": "/data/probe;/data/thruster",
 "PROBE_MCP_OUTPUT": "./probe_results"
 }
 }
  }
}
```

### 命令行测试
```bash
python -c "
from langmuir_mcp.server import probe_analyze
import json
r = probe_analyze('test.csv')
print(json.dumps(r['tail_metrics'], indent=2))
"
```

---

## 9. 后续规划

| 优先级 | 功能 | 说明 |
|--------|------|------|
| P0 | 测试套件 | 用已知 IV 曲线的解析解验证精度 |
| P1 | α 相似参数管线 | 集成 Lafleur & Chabert 2025 约束图 |
| P1 | 双温迭代减法 | Yip 2020 方法作为备选路由 |
| P2 | 束电子 MCC 验证 | 试验粒子 Monte Carlo 交叉验证 |
| P2 | 实时采集模式 | 对接 IT2801 SCPI 协议, 边采边分析 |
| P3 | 多探针支持 | 平面探针/球探针/发射探针 |
| P3 | Web UI | FastAPI + 浏览器直接拖 CSV |

---

## 10. 参考文献

1. Godyak V A, Piejak R B. *Phys. Rev. Lett.* 65, 996 (1990). — T_eff 定义
2. Godyak V A, Demidov V I. *J. Phys. D: Appl. Phys.* 44, 233001 (2011). — 探针 EEDF 综述
3. Jauberteau J L, Jauberteau I. *Contrib. Plasma Phys.* (2018). — f_tail 高能尾积分
4. Lobbia R B, Beal B E. *AIAA JPC* (2017). DOI:10.2514/1.B35531 — 等离子体探针推荐实践
5. Lafleur T, Chabert P. *Plasma Sources Sci. Technol.* 34, 055005 (2025). — α 相似参数
6. Yip C-S et al. *Plasma Sci. Technol.* 22, 085404 (2020). — 双温迭代减法
7. Cellarius R (1970) / Draganov D et al. *Vacuum* 235 (2025). — EEPF 多群分解

---

> **作者**: 许墨 | **实验数据**:  (长光卫星) & 
> **代码**: `E:\长光卫星\claude_api\claude\交付成品_商业MCP\langmuir-probe-mcp\`
