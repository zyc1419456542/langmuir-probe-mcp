# Langmuir Probe MCP v1.2.0

Langmuir probe IV curve analysis MCP Server — pure Python, zero MATLAB dependency.

## Tech Stack

21 MCP tools covering 46/51 analysis techniques, validated on low-temperature plasma experiments:

| Layer | Tools | Techniques |
|-------|-------|------------|
| **L0 Preprocessing** | probe_analyze probe_batch | Gaussian smooth, CSV parse, probe area, batch |
| **L1 Feature Extract** | probe_eepf probe_fit_bimaxwell probe_spectrum | Vf/Vp, Te/Ne, Druyvesteyn EEPF, Yip 2020 bi-Maxwell, FFT |
| **L2 Constraint** | probe_similarity probe_predict_mode | alpha param (Lafleur 2025), mode probability (88.9%) |
| **L3 Mode Detection** | probe_detect_modes | MODE_A / MODE_B / MODE_C classification |
| **L4 Structure** | probe_detect_steps probe_detect_upturn probe_detect_comb_teeth probe_anode_detect | dI/dV staircase, end-upturn, comb-teeth, current jumps |
| **L5 Quality Gate** | probe_gate probe_gate_multi_ai | Numerical/consistency gate + multi-AI audit |
| **L6 Verification** | probe_visual_qa probe_stratify | VL visual check + stratified analysis (Simpson) |
| **L7 Comparison** | probe_compare | Cross-configuration comparison |
| **L8 Visualization** | probe_plot probe_plot_extended | IV+dI/dV, EEPF, OML check, Log(I)-V, bi-Maxwell |
| **L9 Reference** | probe_info | Formulas, ranges, thresholds, bibliography |

## Install

pip install -e .

## Usage

langmuir-probe-mcp

Or configure as MCP server in Claude Code:

{
  "mcpServers": {
    "langmuir-probe": {
      "command": "python",
      "args": ["-m", "langmuir_mcp.server"]
    }
  }
}

## Optional: LLM Enhancement

# Multi-AI audit gate
export PROBE_LLM_NUMERICAL=qwen3:32b
export PROBE_LLM_API_BASE=http://127.0.0.1:11434/v1

# Visual verification
export PROBE_LLM_VISION=llava:13b

## References

- Godyak & Piejak 1990, PRL 65:996 — T_eff definition
- Godyak & Demidov 2011, J. Phys. D 44:233001 — probe EEDF review
- Jauberteau & Jauberteau 2018, Contrib. Plasma Phys. — f_tail
- Lobbia & Beal 2017, AIAA JPC — probe diagnostics best practice
- Lafleur & Chabert 2025, PSST 34:055005 — alpha similarity
- Yip et al. 2020, Plasma Sci. Technol. 22:085404 — bi-Maxwell iteration
- Cellarius 1970 / Draganov 2025, Vacuum 235 — EEPF multi-group

## License

MIT
