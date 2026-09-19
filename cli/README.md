# cli/ — 命令行工具

精细调参用的独立脚本，比 Web 服务暴露更多的开关。日常做证件照用
`./run.sh`（仓库根目录）就够了，这里适合反复试参数。

## 快速开始

```bash

# 一步到底：原图 → 白底成品（不需要先分割、不需要算坐标）
python3 cli/make_id_photo.py "照片.jpg" /tmp/out.jpg --dump-raw /tmp/raw.jpg
python3 cli/retouch.py /tmp/raw.jpg /tmp/retouched.jpg --undereye 0.20

# 降采样到目标尺寸
python3 -c "from PIL import Image; Image.open('/tmp/retouched.jpg').resize((480,640), Image.LANCZOS).save('/tmp/final.jpg','JPEG',quality=92)"
```

## 参数由 Vision 提供，不用手算

所有依赖图像坐标的参数都是**可选**的，不传就自动检测：

| 参数 | 来源 |
|---|---|
| `--crown` 头顶 | 从蒙版轮廓取（首个宽度达到人脸宽 15% 的行，跳过上方游离碎发） |
| `--chin` 下巴 / `--midline` 人脸中线 | Vision 人脸关键点 |
| `--eye` 眼区（`retouch.py`） | Vision 人脸关键点 |

**为什么不留手算路径**：这些是**像素坐标**，和输入图的分辨率、裁切绑定。
一旦图片被重新缩放或裁切，之前抄下来的数字就全部失效 —— 而且不会报错，
只会静默地把眼下提亮、去痣排除区放到错误的位置。需要复现某次结果时再显式传入。

`make_id_photo.py` 的 `mask` 参数同样可选，不传就自己调 Vision 分割。

> ⚠️ 如果你自己传蒙版：**必须和 `src` 同一朝向**。Vision 的分割读原始像素、
> 不认 EXIF，对未归正的 iPhone 原图跑出来的蒙版是横的，和归正后的图对不上。

## 两个脚本的分工

```
make_id_photo.py   分割（可选）→ 蒙版精修 → 人像关键点 → 裁切构图 → 合成底色
        │
        └── --dump-raw 输出全分辨率合成图
                    │
retouch.py ─────────┘  磨皮 / 美白 / 眼下提亮 / 去痣 → 全分辨率成品
```

蒙版精修走的是学习型 matting 模型（见 `idphoto/matting.py`），与服务完全同一套
覆盖率）。传 `--edge-guide` 可回退到早期的导向滤波版本做对比。

## 常用参数

**构图**（`make_id_photo.py`）

```bash
--head-frac 0.64      # 头部占画面高度；0.66 头更大，0.60 更小
--crown-margin 0.09   # 头顶留白
--enhance 1           # 全局提亮（默认关；服务版用的是自适应提亮）
```

**修图**（`retouch.py`）

```bash
--smooth 0.45    # 磨皮力度
--white 2.5      # 美白（L 提升）。调到 7 以上 b* 会掉出正常区间，脸发灰
--red 0.8        # 去红
--yellow 1.5     # 去黄
--undereye 0.20  # 眼袋；超过 0.30 开始显假
--moles 1.0      # 去痣（默认关，痣属身份特征）
```

## 与服务版的关系

两者共用 `idphoto/` 包里的算法（抠图、人脸解析、修图），所以给定同样的输入，
结果一致。差异只来自输入路径：工具若吃一张二次编码过的 JPEG，像素会和直接解码
原图有微小出入，裁切框可能差十几像素。

坐标解析只有**一份**，在 `idphoto/vision.py`。`_vision.py` 只是个路径 shim，
把这唯一一份引进来 —— 之前这里复制过一份，其中 min/max 传错顺序，
静默地把眼区算歪了。
