
# Advanced Face Refiner (AFR) / 高级面部修复器 / 高度な顔修正器

---

## 🌐 English

A Stable Diffusion WebUI / SD Forge Classic extension for advanced face refinement with 2-pass inpainting and MediaPipe FaceMesh based polygon masks.

### Features

- **2-Pass Inpainting**:
  - Pass 1 (low denoising strength): Fix facial structure and alignment issues
  - Pass 2 (normal denoising strength): Refine detailed features like eyes, eyelashes, etc.
- **MediaPipe FaceMesh Polygon Masks**: Uses facial landmarks instead of square YOLO boxes for seamless edge transitions
- **Dedicated UI Panel**: Always visible accordion with separate controls for each pass, face prompts, and detection settings
- **Full Compatibility**: Works seamlessly with both Stable Diffusion WebUI and SD Forge Classic
- **Multi-Language Support**: UI automatically adapts to WebUI language settings (English, 中文, 日本語)

### Installation

1. Download or clone this repository into your `extensions/` folder
2. Restart SD WebUI / SD Forge Classic
3. Dependencies (mediapipe, opencv-python, etc.) will be installed automatically

### Usage

1. Go to **Txt2Img** or **Img2Img** tab
2. Expand the **Advanced Face Refiner (AFR)** accordion at the bottom
3. Check **Enable AFR**
4. Adjust parameters as needed
5. Generate your image

---

## 🌐 中文

适用于 Stable Diffusion WebUI 和 SD Forge Classic 的高级面部修复插件，采用双阶段局部重绘和基于 MediaPipe FaceMesh 的多边形掩码。

### 功能特性

- **双阶段局部重绘**:
  - 第1阶段（低重绘强度）：修复面部结构和对齐问题
  - 第2阶段（正常重绘强度）：精修眼瞳、睫毛等细节
- **MediaPipe FaceMesh 多边形掩码**：使用面部关键点而非传统的方形 YOLO 框，实现无缝边缘过渡
- **专用 UI 面板**：常驻折叠面板，包含各阶段独立控制、面部提示词和检测设置
- **完美兼容**：同时支持 Stable Diffusion WebUI 和 SD Forge Classic
- **多语言支持**：UI 自动适配 WebUI 语言设置（英语、中文、日语）

### 安装

1. 下载或克隆此仓库到 `extensions/` 文件夹
2. 重启 SD WebUI / SD Forge Classic
3. 依赖项（mediapipe、opencv-python 等）会自动安装

### 使用方法

1. 前往 **Txt2Img** 或 **Img2Img** 标签页
2. 展开底部的 **高级面部修复器 (AFR)** 折叠面板
3. 勾选 **启用 AFR**
4. 根据需要调整参数
5. 生成图像

---

## 🌐 日本語

2パスインペイントとMediaPipe FaceMeshベースのポリゴンマスクを使用した、Stable Diffusion WebUI / SD Forge Classic向けの高度な顔修正拡張機能。

### 特徴

- **2パスインペイント**:
  - 第1段階（低ノイズ除去強度）：顔の構造と位置ずれの問題を修正
  - 第2段階（通常のノイズ除去強度）：目、睫毛などの詳細な特徴を精錬
- **MediaPipe FaceMeshポリゴンマスク**：正方形のYOLOボックスの代わりに顔のランドマークを使用し、シームレスなエッジ遷移を実現
- **専用UIパネル**：各パス用の個別コントロール、顔のプロンプト、検出設定を備えた常時表示のアコーディオン
- **完全互換**：Stable Diffusion WebUIとSD Forge Classicの両方でシームレスに動作
- **多言語サポート**：WebUIの言語設定に自動的に適応（英語、中文、日本語）

### インストール

1. このリポジトリをダウンロードまたはクローンして`extensions/`フォルダに配置
2. SD WebUI / SD Forge Classicを再起動
3. 依存関係（mediapipe、opencv-pythonなど）が自動的にインストールされます

### 使用方法

1. **Txt2Img**または**Img2Img**タブに移動
2. 下部の**高度な顔修正器 (AFR)**アコーディオンを展開
3. **AFRを有効化**をチェック
4. 必要に応じてパラメータを調整
5. 画像を生成

---

## 📝 License

MIT

