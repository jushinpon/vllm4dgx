# vllm4dgx

NVIDIA DGX Spark / GB10 平台的 vLLM 安裝、驗證、快照、還原工具集。

本儲存庫提供四個核心腳本，確保在 DGX Spark / GB10（Blackwell/SM100）架構上維持一個「已知可用」的 vLLM 環境。

---

## 為什麼需要這個 repo

在 DGX Spark / GB10 上，上游 `vllm` 安裝容易因以下原因毀壞：

- Triton 版本不匹配
- FlashInfer 封裝/編譯問題
- vLLM CMake 目標在 Blackwell/SM100 上的線程問題
- editable-install import 路徑問題
- 依賴套件漂移

本 repo 透過 `install.sh` 鎖定已知可運作版本，並提供 `vllm_workable_info.pl` 和 `restore_vllm_from_snapshot.pl` 來防止環境漂移。

## 鎖定版本

| 套件 | 版本 |
|---|---|
| vLLM | `66a168a197ba214a5b70a74fa2e713c9eeb3251a` |
| Triton | `4caa0328bf8df64896dd5f6fb9df41b0eb2e750a` (v3.5.0) |
| Python | 3.12 |
| PyTorch | cu130 |

安裝時 `install.sh` 會修正 `CMakeLists.txt` 以支援 Blackwell 架構，並新增 `grouped_mm_c3x_sm100.cu` 至 `_C` 擴展目標。

---

## 檔案清單

| 檔案 | 功能 |
|---|---|
| `install.sh` | 一鍵安裝已知可用的 vLLM 環境 |
| `smoke_test_vllm_model.sh` | 安裝後煙霧測試 + vLLM API 伺服器啟動 |
| `vllm_workable_info.pl` | 快照當前可用環境，比較未來安裝 |
| `restore_vllm_from_snapshot.pl` | 從快照還原/修復漂移的安裝 |
| `installation_check_cmd.txt` | 安裝檢查命令參考 |

---

## 依賴環境

| 項目 | 需求 |
|---|---|
| OS | Rocky Linux 8/9（或相容 RHEL） |
| GPU | NVIDIA DGX Spark / GB10（Blackwell, aarch64） |
| 驅動 | NVIDIA Driver 580.x+ |
| CUDA | 13.x（含 nvcc） |
| Python | 3.12 |
| 工具 | git, curl, uv（若未安裝會自動安裝） |
| 權限 | root（預設安裝至 /local_opt） |

---

## 使用方法

### 1. 全新安裝

```bash
deactivate 2>/dev/null || true
rm -rf /local_opt/vllm-install
bash install.sh --install-dir /local_opt/vllm-install |& tee /home/install.log
```

**參數：**

| 參數 | 說明 | 預設值 |
|---|---|---|
| `--install-dir DIR` | 安裝目錄 | 當前目錄下的 `vllm-install` |
| `--vllm-version HASH` | vLLM git commit | `66a168a197ba214a5b70a74fa2e713c9eeb3251a` |
| `--python-version VER` | Python 版本 | `3.12` |

安裝過程約 20-30 分鐘（CPU 編譯）。

### 2. 煙霧測試

```bash
# 基本測試
bash smoke_test_vllm_model.sh --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct

# 測試並保持伺服器運行
bash smoke_test_vllm_model.sh \
  --stop-existing \
  --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
  --test-chat \
  --keep-server

# 低記憶體測試
bash smoke_test_vllm_model.sh \
  --stop-existing \
  --model /local_opt/vllm-models/Qwen-Qwen2.5-7B-Instruct \
  --gpu-memory-util 0.15 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 256 \
  --enforce-eager \
  --test-chat
```

**參數：**

| 參數 | 說明 | 預設值 |
|---|---|---|
| `--model PATH_OR_HF_ID` | 模型路徑或 HF ID | **必填** |
| `--install-dir DIR` | vLLM 安裝目錄 | `/local_opt/vllm-install` |
| `--host HOST` | API 伺服器 host | `127.0.0.1` |
| `--port PORT` | API 伺服器 port | `8000` |
| `--gpu-memory-util FLOAT` | GPU 記憶體使用比例 | `0.30` |
| `--max-model-len INT` | 最大上下文長度 | `4096` |
| `--max-num-seqs INT` | 最大並行序列 | `1` |
| `--max-num-batched-tokens INT` | 最大批次 token | `1024` |
| `--test-chat` | 執行聊天測試 | 關閉 |
| `--keep-server` | 測試後保留伺服器 | 關閉 |
| `--stop-existing` | 先停止現有 vLLM | 關閉 |
| `--enforce-eager` | 停用 CUDA graph 以減低記憶體 | 關閉 |
| `--max-wait SEC` | 伺服器啟動等待秒數 | `600` |

### 3. 環境快照

```bash
# 建立當前環境快照
perl vllm_workable_info.pl snapshot

# 自訂路徑
perl vllm_workable_info.pl snapshot \
  --install-dir /local_opt/vllm-install \
  --out-dir /local_opt/workable_llm_info
```

**快照輸出（/local_opt/workable_llm_info/）：**

| 檔案 | 內容 |
|---|---|
| `snapshot_meta.txt` | 快照時間、路徑、vLLM SO 位置 |
| `requirements_freeze.txt` | pip freeze 套件清單 |
| `env_summary.txt` | torch/triton/flashinfer/vllm 版本 JSON |
| `api_server_help.txt` | `api_server --help` 輸出 |
| `symbol_check.txt` | `_C.abi3.so` 中的 `cutlass_moe_mm_sm100` 符號 |
| `CMakeLists.final_working.txt` | 當前 CMake 配置備份 |

### 4. 環境比較

```bash
# 比較新安裝與快照
perl vllm_workable_info.pl compare --target /path/to/new-install
```

輸出包含：
- 套件 freeze diff
- 環境摘要 diff
- 符號檢查 diff
- 快速狀態報告（triton 版本、CUDA 可用、符號定義等）

### 5. 從快照還原

```bash
# 還原至已知可用狀態
perl restore_vllm_from_snapshot.pl

# 預覽（不執行）
perl restore_vllm_from_snapshot.pl --dry-run
```

**還原流程：**

1. 顯示當前環境狀態
2. 強制重裝鎖定套件（numpy==2.2.6, transformers==4.56.0, tokenizers==0.22.2）
3. 檢查/修復 CMake patch（SM100 支援）
4. 重裝 Triton 3.5.0
5. 重編譯 vLLM
6. 最終驗證

---

## 輸入/輸出格式

### install.sh 輸出

```
========================================
Creating virtual environment
========================================
[INFO] Install directory : /local_opt/vllm-install
[INFO] vLLM commit       : 66a168a197ba214a5b70a74fa2e713c9eeb3251a
[INFO] Triton commit     : 4caa0328bf8df64896dd5f6fb9df41b0eb2e750a
...
[SUCCESS] Installation verified
```

### smoke_test_vllm_model.sh 輸出

```
[PASS] /v1/models responded
{"data": [{"id": "model-name", ...}]}
[PASS] Chat test completed
[PASS] Smoke test passed
```

### vllm_workable_info.pl 輸出

```
[INFO] Creating snapshot from: /local_opt/vllm-install
[INFO] Saving into: /local_opt/workable_llm_info
[PASS] Snapshot completed.
```

### restore_vllm_from_snapshot.pl 輸出

```
========== vLLM Restore From Snapshot ==========
[INFO] install_dir = /local_opt/vllm-install
========== Fix Python package pins ==========
...
[PASS] api_server --help works
[PASS] cutlass_moe_mm_sm100 is defined
[PASS] environment restored
[PASS] Restore procedure finished.
```

---

## AI Agent 操控指南

### 當你需要驗證 vLLM 環境是否正常時

```
任務: 檢查 vLLM 環境
步驟:
1. 執行: perl vllm_workable_info.pl compare --target /local_opt/vllm-install
2. 檢查輸出中的 [OK] / [NO] 狀態
3. 若發現 [NO]，執行: perl restore_vllm_from_snapshot.pl --dry-run 預覽修復步驟
4. 確認後執行: perl restore_vllm_from_snapshot.pl 實際修復
5. 修復後再次 compare 驗證
```

### 當你需要在新節點安裝 vLLM 時

```
任務: 新節點 vLLM 安裝
步驟:
1. 確認節點有 NVIDIA driver + CUDA + nvidia-smi
2. 確認 /local_opt 有足夠空間（至少 20GB）
3. 複製 repo 到節點
4. 執行: bash install.sh --install-dir /local_opt/vllm-install |& tee /home/install.log
5. 安裝完成後: perl vllm_workable_info.pl snapshot 建立快照
6. 執行: bash smoke_test_vllm_model.sh --model <MODEL_PATH> --test-chat 驗證
```

### 當環境出現問題時（依賴漂移）

```
任務: 修復漂移的 vLLM 環境
步驟:
1. 執行: perl vllm_workable_info.pl compare --target /local_opt/vllm-install
2. 若 diff 顯示版本變化:
   a. perl restore_vllm_from_snapshot.pl --dry-run（預覽）
   b. perl restore_vllm_from_snapshot.pl（實際修復）
3. 驗證: bash smoke_test_vllm_model.sh --model <MODEL_PATH> --test-chat
```

### 疑難排解

| 症狀 | 原因 | 解決方式 |
|---|---|---|
| `cutlass_moe_mm_sm100 is not defined` | CMake patch 遺失 | `perl restore_vllm_from_snapshot.pl` |
| `Triton version != 3.5.0` | Triton 被依賴重裝覆蓋 | `perl restore_vllm_from_snapshot.pl` |
| `vllm._C.abi3.so not found` | vLLM 未編譯 | 重跑 `install.sh` |
| `CUDA not available` | PyTorch 未正確安裝 | 檢查 nvcc 和 CUDA 驅動 |
| 伺服器啟動失敗 | 記憶體不足 | 用 `--gpu-memory-util 0.15` 降低 |
