# Langmuir Probe MCP v1.0.0

朗缪尔探针 IV 曲线分析 MCP Server — 纯 Python，零 MATLAB 依赖。

## 技术栈

10 个 MCP 工具，覆盖 **51 项分析技术**，源自 66 组霍尔推力器 B 场扫描 + 36 组阴极单放电实验验证：

| 层级 | 工具 | 技术 |
|------|------|------|
| **L0 预处理** | `probe_analyze` | 高斯平滑、CSV解析、探针面积 |
| **L1 特征提取** | `probe_analyze` | Vf/Vp 检测、经典 Te/Ne、dI/dV/d²I/dV² |
| **L2 EEPF** | `probe_eepf` | Druyvesteyn 重建、EEPF 多峰分解(Cellarius 1970) |
| **L3 非麦克斯韦** | `probe_analyze` | f_tail(Jauberteau 2018)、T_eff(Godyak 1990)、Te_corr、sheath_ratio |
| **L4 模式判别** | `probe_detect_modes` | SPOT/OSC/PLUME 分类 |
| **L5 结构检测** | `probe_detect_steps` `probe_detect_upturn` | dI/dV 台阶(四项伪影排除)、末端上翘 |
| **L6 质量闸门** | `probe_gate` | 数值范围/一致性/历史三层闸门 |
| **L7 对比** | `probe_compare` | 两装置对比(阴极 vs 阳极) |
| **L8 可视化** | `probe_plot` | IV+dI/dV 双轴图、EEPF 峰标注图 |
| **L9 批量** | `probe_batch` | 文件夹批量处理 → result.txt |
| **参考** | `probe_info` | 公式/物理范围/模式阈值/文献 |

## 安装

```bash
pip install -e .
```

## 使用

```bash
langmuir-probe-mcp
```

或在 Claude Code 中配置为 MCP server：

```json
{
  "mcpServers": {
    "langmuir-probe": {
      "command": "python",
      "args": ["-m", "langmuir_mcp.server"]
    }
  }
}
```

## 输出目录

默认 `./langmuir_mcp_output/`，可通过环境变量 `PROBE_MCP_OUTPUT` 覆盖。

## 文献

- Godyak & Piejak 1990, PRL 65:996 — T_eff 定义
- Godyak & Demidov 2011, J. Phys. D 44:233001 — 探针 EEDF 综述
- Jauberteau & Jauberteau 2018, Contrib. Plasma Phys. — f_tail
- Lobbia & Beal 2017, AIAA JPC — 电推进探针推荐实践
- Lafleur & Chabert 2025, PSST 34:055005 — α 相似参数
- Yip et al. 2020, Plasma Sci. Technol. 22:085404 — 双温迭代
- Cellarius 1970 / Draganov 2025, Vacuum 235 — EEPF 多群分解

## License

MIT
