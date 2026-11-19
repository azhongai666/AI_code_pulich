# AI_code_pulich

> AI 学习与科研实战代码库 —— Google Colab 入门、深度学习医学图像分类、YOLO26 高分辨率切片预测、数据集平台横向对比。
> 全部内容以可直接运行的 `.ipynb` / 单页 `.html` 形式提供，无需配置本地环境。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-0b23a9)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-f9ab00)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 一、总览

### 1.1 这个仓库是什么

一个**按主题划分的教程型代码仓库**，四份主文件对应四条独立学习线：

| 主线 | 回答的问题 | 载体 |
| :--- | :--- | :--- |
| **环境线** | 没有 GPU、没有服务器，怎么免费跑深度学习？ | Colab 入门 Notebook |
| **医学影像线** | 拿到一个 CT 数据集，怎么从零走完"读数据 → Dataset → DataLoader"？ | 医学图像分类·第 2 章 Notebook |
| **数据源线** | Kaggle / Roboflow / 天池 / GitHub / HF，我该用哪个？ | 单页对比 HTML |
| **检测线** | 大图里的小目标全被缩没了，怎么办？ | YOLO26 切片预测 Notebook |

四份内容互相独立，可单独取用；也可以按"环境 → 数据 → 模型"的顺序串成一条入门路径（见 [§4 推荐学习路径](#四推荐学习路径)）。

### 1.2 文件明细总表

> 时间均为**最后提交时间**（北京时间 UTC+8），取自仓库 commit 记录；仓库创建于 2025-09-20。

| # | 文件 | 类型 | 体积 | 上传 / 最后修改时间 | 内容概要 |
| :-: | :--- | :--- | ---: | :--- | :--- |
| 1 | `10分钟掌握Google_Colab_最强薅羊毛！白嫖GPU！AI入门服务器！___阿钟AI.ipynb` | Notebook | 2.35 MB | **2025-09-24 11:45** | Colab 全景入门：界面与快捷键、开启 GPU、装库、传数据/挂 Drive、PyTorch 跑通 MNIST、中文字体、Shell 技巧、Gemini 调用。57 个单元格（28 说明 + 29 代码） |
| 2 | `一周搞定AI毕设_深度学习医学图像分类_第2章_数据集准备与预处理___阿钟AI.ipynb` | Notebook | 2.08 MB | **2025-09-27 11:22** | 医学图像分类系列**第 2 章**：IQ-OTHNCCD 肺癌 CT 数据集（1097 张 / 3 类）的统计分析、预处理、自定义 Dataset、DataLoader、三类可视化。44 个单元格（23 + 21） |
| 3 | `261109_使用YOLO26_进行高分辨率切片预测.ipynb` | Notebook | 6.43 MB | **2026-11-09 20:28** | Ultralytics YOLO26 的 `tile` 切片推理：原理、整图 vs 切片同帧对比（45 → 102 个检测）、视频流式预测、CLI 等价写法、`track`/`train`/`val` 的切片用法。25 个单元格（14 + 11） |
| 4 | `深度学习数据集来源详细对比.html` | 单页 HTML | 35.2 KB | **2025-09-20 16:43** | 5 大平台（Kaggle / Roboflow / 天池 / GitHub / Hugging Face）× 11 个维度的对比表，附 6 类人群的选型建议与 6 条使用原则 |
| 5 | `.gitignore` | 配置 | 4.6 KB | **2025-09-20 16:08** | Python 标准忽略模板（`__pycache__`、`*.py[codz]`、`build/`、`dist/`、虚拟环境等），与初始提交一同入库 |
| 6 | `LICENSE` | 协议 | 1.0 KB | **2025-09-20 16:08** | MIT License，Copyright (c) 2025 azhongai666 |
| 7 | `README.md` | 文档 | — | 本次重写 | 你正在看的这份文件（原内容仅一行标题 `# AI_code_pulich`，16 字节） |

**仓库规模**：7 个文件（无子目录）、约 8 MB，主语言 `Jupyter Notebook`，默认分支 `main`。

### 1.3 时间线

```
2025-09-20  16:08  初始提交 c132919  ── README / LICENSE / .gitignore
2025-09-20  16:43  提交 0f010bc     ── 数据集来源对比 HTML
2025-09-24  11:45  提交 40ca86c     ── Colab 教程 Notebook
2025-09-27  11:22  提交 978dda4     ── 医学图像分类 第 2 章
2026-11-09  20:28  提交 189cab8     ── YOLO26 切片预测（Colab 直建）
```

### 1.4 命名约定

| 前缀 / 后缀 | 含义 | 示例 |
| :--- | :--- | :--- |
| `YYMMDD_` | 制作日期，便于按时间排序 | `261109_` → 2026-11-09 |
| `___阿钟AI` | 作者署名后缀 | `...___阿钟AI.ipynb` |
| `一周搞定AI毕设_` | 系列课编号（本仓库只放了第 2 章） | 系列共 10 章，目录见该 Notebook 开篇 |
| `.html` | 无需运行、双击即看的可视化图表 | 数据集对比页 |

---

## 二、分述：逐个文件详解

### 2.1 Colab 入门 Notebook

**文件**：`10分钟掌握Google_Colab_最强薅羊毛！白嫖GPU！AI入门服务器！___阿钟AI.ipynb`
**规模**：57 cells = 28 markdown + 29 code，含 5 张输出图

一份"从打开网页到跑出准确率"的完整 Colab 速通，章节结构：

| 章节 | 内容要点 |
| :--- | :--- |
| 一、什么是 Colab | 免费 GPU 的原理与额度限制 |
| 二、快速开始 | 新建 Notebook、运行第一行代码 |
| 三、基础操作 | 代码单元格 vs 文本单元格、快捷键速查 |
| 四、启用 GPU | `!nvidia-smi` 验证显卡是否挂上 |
| 五、安装与导入库 | 预装库清单；`!pip install ultralytics==8.3.202`、`opencv-python`、`transformers` |
| 六、数据管理 | 上传本地文件、`drive.mount('/content/drive')`、`files.download()` 取回结果、`!wget` 直下数据到服务器、`!zip -r` 打包 |
| 七、第一个深度学习示例 | PyTorch MNIST：`SimpleNN`（784 → 128 → 10，Flatten + FC + ReLU + FC）、Adam(lr=1e-3)、batch=64、快速演示先跑约 200 个 batch |
| 八、实用技巧 | `!cat /proc/cpuinfo`、`!free -h`、`!ls -lh`、`!pwd` 等系统探查 |
| 九～十一 | 注意事项、学习资源、Gemini AI 用法 |

**核心代码骨架**（第七节）：

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(self.flatten(x))))

model = SimpleNN().to("cuda" if torch.cuda.is_available() else "cpu")
criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)
```

> 说明：训练循环设了 `if batch_idx > 200: break`，是为"10 分钟演示"服务的**故意截断**，不是完整训练，别把这条曲线的 loss 当收敛结果看。

---

### 2.2 医学图像分类 · 第 2 章（数据集准备与预处理）

**文件**：`一周搞定AI毕设_深度学习医学图像分类_第2章_数据集准备与预处理___阿钟AI.ipynb`
**规模**：44 cells = 23 markdown + 21 code，含 5 张输出图，环境记录为 Python 3.8.5

开篇先给出**整个系列的 10 章目录**（本文件只实现第 2 章，其余章节未入库）：

```
第1章 Colab 环境 → 第2章 数据集准备 ← 本文件 → 第3章 第一个 CNN
→ 第4章 迁移学习 → 第5章 评估/混淆矩阵 → 第6章 Gradio 界面
→ 第7章 降维可视化 → 第8章 Grad-CAM 可解释性 → 第9章 ONNX 部署
→ 第10章 Gemini API 图像识别（附录）
```

**用的数据**：Kaggle `adityamahimkar/iqothnccd-lung-cancer-dataset`（IQ-OTHNCCD 肺癌 CT，约 198 MB）。Notebook 里跑出的真实统计：

| 项目 | 实际输出 |
| :--- | :--- |
| 类别数 | 3：`Malignant cases` / `Normal cases` / `Benign cases`（原数据集目录拼作 `Bengin cases`，非笔误） |
| 图像总数 | 1097 张（Malignant 561、Normal 416，余下为 Benign） |
| 训练 / 验证划分 | 877 / 220（8 : 2） |

**本文件实际做了什么**：

1. **取数**：`!curl -L -o ...zip` 直连 Kaggle API 下载 → `!unzip` 解压（绕过需登录的 `kaggle` CLI）。
2. **中文显示**：下载 `SimHei.ttf` 并 `fm.fontManager.addfont()`，否则所有 `plt.title` 中文全是方块。
3. **数据集体检**：`analyze_dataset()` 自动扫子文件夹拿类别名，`count_images_per_class()` 统计各类数量 → 暴露类别不均衡。
4. **预处理**：统一 `IMG_SIZE=(224,224)`（ImageNet 惯例）；`train_transform` 走增强，验证集只做 Resize + ToTensor + Normalize。
5. **自定义 Dataset**：`ImageClassificationDataset`，实现 `__init__ / __len__ / __getitem__`，自动建 `class_to_idx` / `idx_to_class` 双向映射，支持 `.jpg/.jpeg/.png/.bmp/.tiff`，带路径不存在等异常兜底。
6. **DataLoader**：`BATCH_SIZE=32`、`NUM_WORKERS=4`、训练 `shuffle=True` / 验证 `False`，用 `torch.utils.data.Subset` 做 8:2 划分。
7. **可视化三件套**：`show_sample_images()` 随机 4×2 网格看样本（含反标准化还原像素）、`plot_class_distribution()` 类别分布柱状图、`show_augmentation_effects()` 原图 vs 水平翻转 / 随机旋转 30° / 亮度调整 对比。
8. **收尾**：一段总结 print + "后续数据处理 5 个方向"（数据清洗、AutoAugment/RandAugment、类别不平衡（加权采样/SMOTE/焦点损失）、GPU 端在线增强与缓存、数据版本管理与血缘追踪）。

```python
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

> 跑之前必改一处：`data_root` 硬编码为 `/content/The IQ-OTHNCCD lung cancer dataset/...`，换机器/换数据集要同步改路径，否则后续全链路直接空转（代码有判空，不会报错，只会静默跳过）。

---

### 2.3 深度学习数据集来源详细对比（HTML）

**文件**：`深度学习数据集来源详细对比.html`（35.2 KB，单文件、零依赖、浏览器直接打开）

一张主表 + 两段建议：

- **主表**：`Kaggle` / `Roboflow` / `天池` / `GitHub` / `Hugging Face` × 11 个维度 —— 平台定位、初学者友好度（★ 评级）、数据集类型、数据质量、使用成本、下载方式、社区支持、学习资源、优点、缺点、最适合的初学者类型。
- **场景选型**：刚开始学 → Kaggle/HF；专注 CV → Roboflow；中文环境 → 天池；NLP → HF；偏工程落地 → Kaggle + Roboflow；追学术 → HF + Kaggle 竞赛方案。
- **6 条使用原则**：循序渐进、质量优先、积极参与社区、注重文档、实践为主、注意版权。

适合发给"第一步该去哪找数据"的新人，也可以作为课程第一课的讲义截图。

---

### 2.4 YOLO26 高分辨率切片预测

**文件**：`261109_使用YOLO26_进行高分辨率切片预测.ipynb`
**规模**：25 cells = 14 markdown + 11 code，体积 6.43 MB（大图以 base64 内嵌），由 Colab 直接创建

**要解决的问题**：YOLO 会把图像最长边压到 `imgsz`（默认 640）。一张 4K 航拍图里 20 像素的小车，压完只剩 3 个像素——直接消失。`tile` 参数的做法是把大图切成互叠 20% 的窗口，逐窗推理后再把框合并回原图坐标。

**逐步走完的流程**：

| 步骤 | 代码 / 结论 |
| :--- | :--- |
| 装环境 | `!uv pip install ultralytics` + `ultralytics.checks()` |
| 取数据 | `safe_download(...)` 下载 Ultralytics 官方环岛航拍视频 `roundabout.mp4`（960×1080） |
| 同帧双跑 | `model.predict(frame)` vs `model.predict(frame, tile=400)` |
| **效果对比** | 整图 **45** 个检测 → 切片 **102** 个；左上角停车场 **6 → 31** |
| 视频流式 | `predict(..., tile=400, stream=True, save=True)`，逐帧限制显存；`tile=400` 时每帧拆 12 个窗口，务必用 GPU 运行时 |
| CLI 等价 | `yolo predict model=yolo26s.pt source=roundabout.mp4 tile=400 save=True` |
| 跟踪 | `model.track("roundabout.mp4", tile=400)`，跟踪器在合并后的结果上跑一次 |
| 训练/验证 | `model.train(data="VisDrone.yaml", tile=True, imgsz=640, epochs=100)`；验证先合并切片再对原标注打分（VisDrone 约 2 GB，示例**未实跑**） |

**参数要点**（正文第 2 格原文归纳）：

- `tile=True` → 用 `imgsz` 当窗口边长；`tile=400` → 单独指定窗口尺寸。
- 小于 `imgsz` 的窗口会被**放大**到模型输入尺寸，小目标因此变得可检出（`tile=400` ≈ 放大 1.6×）。
- **代价**：每个窗口一次额外前向，算力线性上涨；窗口过小还会把背景杂波放大成误报。所以作者的建议是"在自己的数据上对比切片与整图，再决定用不用"。

---

### 2.5 工程杂项

| 文件 | 说明 |
| :--- | :--- |
| `.gitignore` | GitHub 官方 Python 模板（约 4.6 KB）：`__pycache__/`、`*.py[codz]`、`build/`、`dist/`、`*.so`、虚拟环境目录等。Notebook 里下载的 `.zip` / 数据集 / 权重**没有**忽略规则，注意别误提交大文件 |
| `LICENSE` | MIT License，Copyright (c) 2025 azhongai666。可自由使用、修改、商用，保留版权声明即可 |
| `README.md` | 原为 16 字节空壳（仅 `# AI_code_pulich`），本次重写为本文件 |

---

## 三、快速开始

三个 Notebook 都自带 `Open in Colab` 徽章，点开即可运行，无需 clone：

| Notebook | 一键打开 |
| :--- | :--- |
| Colab 入门 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azhongai666/AI_code_pulich/blob/main/10%E5%88%86%E9%92%9F%E6%8E%8C%E6%8F%A1Google_Colab_%E6%9C%80%E5%BC%BA%E8%96%85%E7%BE%8A%E6%AF%9B%EF%BC%81%E7%99%BD%E5%AB%96GPU%EF%BC%81AI%E5%85%A5%E9%97%A8%E6%9C%8D%E5%8A%A1%E5%99%A8%EF%BC%81___%E9%98%BF%E9%92%9FAI.ipynb) |
| 医学图像 第2章 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azhongai666/AI_code_pulich/blob/main/%E4%B8%80%E5%91%A8%E6%90%9E%E5%AE%9AAI%E6%AF%95%E8%AE%BE_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%8C%BB%E5%AD%A6%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB_%E7%AC%AC2%E7%AB%A0_%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86___%E9%98%BF%E9%92%9FAI.ipynb) |
| YOLO26 切片 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azhongai666/AI_code_pulich/blob/main/261109_%E4%BD%BF%E7%94%A8YOLO26_%E8%BF%9B%E8%A1%8C%E9%AB%98%E5%88%86%E8%BE%A8%E7%8E%87%E5%88%87%E7%89%87%E9%A2%84%E6%B5%8B.ipynb) |

本地运行（可选）：

```bash
git clone https://github.com/azhongai666/AI_code_pulich.git
cd AI_code_pulich
pip install torch torchvision ultralytics seaborn opencv-python jupyter
jupyter notebook    # 或 jupyter lab
```

**环境与依赖一览**

| 依赖 | 用在哪 | 备注 |
| :--- | :--- | :--- |
| `torch` / `torchvision` | 2.1、2.2 | Colab 预装，CUDA 版本随运行时变化 |
| `ultralytics` | 2.1（8.3.202）、2.4（最新） | 2.4 用 `uv pip install`，版本浮动可能影响 `tile` 参数行为 |
| `matplotlib` / `seaborn` | 全部 | 中文需手动注入 `SimHei.ttf` |
| `opencv-python` | 2.1、2.4 | 视频抽帧 `cv2.VideoCapture` |
| `pandas` / `numpy` | 2.1 | — |
| `google.colab` | 2.1、2.2 | 仅 Colab 环境可用；本地跑需注释掉 `drive.mount` / `files.download` |

---

## 四、推荐学习路径

```
① Colab 入门 Notebook        ← 先把"能跑起来"这件事解决（30 min）
        ↓
② 数据集对比 HTML             ← 知道自己该去哪找数据（10 min 阅读）
        ↓
③ 医学图像 第 2 章            ← 把数据变成 Dataset / DataLoader（1~2 h）
        ↓
④ YOLO26 切片预测             ← 进入检测任务，理解 imgsz / tile 的取舍（30 min）
```

若只想解决单一问题，直接跳到对应文件即可，四份内容没有代码耦合。

---

## 五、常见坑

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| `nvidia-smi` 报错 / GPU 不可用 | 运行时仍是 CPU | 菜单「代码执行程序 → 更改运行时类型 → T4 GPU」 |
| 图表中文全是方块 | matplotlib 无中文字体 | 跑第 2 章的中文字体 cell（下载 `SimHei.ttf` + `addfont`） |
| `data_root` 不存在，后续静默跳过 | 路径写死在 `/content/...` | 改成自己的解压路径；上传后 `!ls` 确认目录层级（数据集常套两层同名文件夹） |
| Kaggle 下载 403 | 没走 `curl` 直连或 token 失效 | 用 Notebook 里的 `https://www.kaggle.com/api/v1/datasets/download/...` 直链 |
| 切片推理反而更慢/误检变多 | 窗口数线性增加算力；小窗口放大背景噪声 | 先在小样本上对比整图与切片的 mAP/检测数，再定 `tile` 值 |
| 训练 loss 降不下来（第 2 章） | 只到"数据准备"，还没建模型 | 等第 3 章；本文件不负责训练 |

---

## 六、作者与联系

| 渠道 | 账号 |
| :--- | :--- |
| 小红书 | 阿钟AI论文辅导 |
| B 站 | 阿钟AI算法 |
| QQ | 2084339727 |
| GitHub | [@azhongai666](https://github.com/azhongai666) |

---

## 七、License

本项目基于 [MIT License](LICENSE) 开源，Copyright (c) 2025 azhongai666。
Notebook 中引用的第三方资源（IQ-OTHNCCD 数据集、Ultralytics 示例视频与模型权重）版权归各自作者所有，商用前请核对原始许可。

---

<sub>📌 仓库内容为教学演示代码，医学影像部分不构成任何诊断建议。</sub>
