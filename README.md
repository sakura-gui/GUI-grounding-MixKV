##  Quick Start

Use the shell scripts to launch the evaluation. The evaluation setup follows the same protocol as **ScreenSpot**, including data format, annotation structure, and metric calculation.Qwen2.5vl and uivenus-7b are both supported.

### Grounding
- **For 7B model:**
  ```bash 
  bash scripts/run_gd_7b.sh
  ```
  Remember to use your own  data path and  model path . Screenspot-v2 is supported now
  the system path in UI-Venus/models/grounding/ui_venus_ground_7b.py  should also be changed 


