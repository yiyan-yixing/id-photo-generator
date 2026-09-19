# 抠图模型

模型文件**不在仓库里**（BiRefNet 单个 928 MB，超过 GitHub 100 MB 单文件上限，
也不该把 1 GB 二进制塞进 git 历史）。首次运行 `./run.sh` 会自动下载，
或手动执行：

```bash
python3 scripts/fetch_models.py            # 两个都下
python3 scripts/fetch_models.py birefnet   # 只下高精度那个
python3 scripts/fetch_models.py rmbg       # 只下快速那个
```

## 需要哪些、放到哪

脚本会把文件放到**本目录**（`vendor/models/`），文件名必须完全一致：

| 文件名 | 大小 | 用途 | 对应页面上哪一档 |
|---|---|---|---|
| `birefnet.onnx` | 928 MB | BiRefNet，抠图质量最好，细碎飞发保留完整 | 高精度（约 31 秒） |
| `rmbg14.onnx` | 168 MB | RMBG-1.4，快很多，细节略逊 | 标准（约 9 秒） |

```
vendor/models/
├── birefnet.onnx     ← 必需（默认精度档）
└── rmbg14.onnx       ← 可选（只跑高精度档就可以不要）
```

## 从哪里下

| 模型 | 上游仓库 | 本仓库实际使用的导出 |
|---|---|---|
| BiRefNet | [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) | [`onnx-community/BiRefNet-ONNX`](https://huggingface.co/onnx-community/BiRefNet-ONNX) → `onnx/model.onnx` |
| RMBG-1.4 | [`briaai/RMBG-1.4`](https://huggingface.co/briaai/RMBG-1.4) | 同仓库 → `onnx/model.onnx` |

**国内网络**：HuggingFace 主站通常不通，用镜像。脚本默认就优先走镜像，
也可以显式指定：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 scripts/fetch_models.py
```

注意 **GitHub Release 的模型资源在部分网络下也不通**（实测返回 000），
所以走 HuggingFace / hf-mirror，不要走 GitHub。

手动下载时，把文件重命名成上表的文件名放进本目录即可：

```bash
# 例：手动下载 BiRefNet
curl -L -o vendor/models/birefnet.onnx \
  https://hf-mirror.com/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model.onnx
```

## 许可证（重要）

| 模型 | 许可 | 能否商用 |
|---|---|---|
| **BiRefNet** | **MIT** | ✅ 可以，无附加限制 |
| **RMBG-1.4** | BRIA RMBG 1.4 许可（source-available） | ❌ **仅限非商用** |

BiRefNet 的 MIT 许可覆盖上游模型与 onnx-community 的导出，本仓库默认使用它，
所以**默认配置可以商用**。

RMBG-1.4 是 BRIA 的自有许可，模型卡明确写的是
"source-available model for **non-commercial use**"。它只服务于「标准（快）」
这一档，是**可选项**：

- 个人 / 学习 / 非商业用途 —— 可以下，没问题
- **要商用** —— 不要下 `rmbg14.onnx`，只留 `birefnet.onnx`；页面上的「标准」档
  会自动消失（服务检测不到模型就不提供该档），不影响使用

删掉即可，不需要改代码：

```bash
rm vendor/models/rmbg14.onnx
```

## 两个模型的差异不止速度

除了体积和耗时，它们的**输出语义不一样**，接错会得到看起来自信但完全错误的结果：

| | 输入归一化 | 输出 |
|---|---|---|
| BiRefNet | ImageNet mean/std | 原始分数，**需要 sigmoid** |
| RMBG-1.4 | `x - 0.5` | 已是概率，做 min-max 拉伸 |

把 BiRefNet 的输出按 RMBG 的方式用 min-max 处理，结果**只抠出脸部皮肤，
头发和衣服全丢** —— 这个坑实际踩过，所以两种约定写在 `idphoto/matting.py`
的 `BACKENDS` 表里，不是散落在调用处。

## 模型缺失时会怎样

- `birefnet.onnx` 和 `rmbg14.onnx` **都没有** → 服务自动退回 macOS Vision
  系统分割，出图但仍可用，结果里会带一条提示
- 只有 `birefnet.onnx` → 只提供「高精度」档
- 两个都在 → 两档都提供

判断逻辑在 `idphoto/matting.py:available()`，按后端分别检测。

## 磁盘占用

两个模型合计约 **1.1 GB**。只想留一个的话，删掉 `rmbg14.onnx` 省 168 MB
（但失去快速档），或删掉 `birefnet.onnx` 省 928 MB（但边缘质量明显下降，
不推荐）。
