# くじらもなかバナー生成実験

このリポジトリには、「くじらもなか」という商品のプロモーションバナー画像を生成し、分析する実験のコードが含まれています。

## プロジェクト概要

本実験は、AI画像生成モデル（Gemini 3 Pro Image）を使用して、異なるペルソナや使用シーンに応じたプロモーションバナー画像を生成し、その視覚的特徴を定量的に分析することを目的としています。

### 研究目的

- ペルソナ情報（フォーマリティレベル、ライフステージ）に基づいた画像生成の実現
- 生成画像の埋め込みベクトル分析による視覚的特徴の定量化
- 線形分類とクラスタリング分析による画像の分類可能性の評価

## 実験概要

本実験では、Gemini 3 Pro Image APIを使用して、以下の2つの軸に沿ってバナー画像を生成します：

1. **フォーマリティ軸（Formality Axis）**: 贈答シーンのフォーマリティレベル（Level 1-5）
   - Level 1: Private Everyday Use（私的日常使用）
   - Level 2: Personal Treat / Self-Gift（個人的なご褒美）
   - Level 3: Casual Gift for Family / Close Friends（家族・親しい友人への贈り物）
   - Level 4: Formal Gift for Workplace / Social Relations（職場・社会的関係への正式な贈り物）
   - Level 5: Ritual / Ceremonial Gift（儀式・儀礼的な贈り物）

2. **ライフステージ軸（Lifestage Axis）**: ライフステージペルソナレベル（Level 1-5）
   - Level 1: Children（0-12歳）
   - Level 2: Youth / Students（13-25歳）
   - Level 3: Early Working Adults（26-35歳）
   - Level 4: Family & Established Adults（35-60歳）
   - Level 5: Seniors（60歳以上）

生成された画像は、SigLIP 2モデルで埋め込みベクトルに変換され、線形分類とPCA可視化により分析されます。

### 使用技術

- **画像生成**: Gemini 3 Pro Image Preview API (`gemini-3-pro-image-preview`)
- **画像埋め込み**: SigLIP 2モデル (`google/siglip2-so400m-patch16-384`)
- **分析**: scikit-learn（線形分類、PCA、クラスタリング評価）
- **可視化**: matplotlib, seaborn

## 必要な環境

- **Google Colab**（推奨）またはPython 3.12+
- GPU推奨（SigLIP 2モデルの実行のため）

## セットアップ

### 1. APIキーの準備

Gemini APIキーが必要です。以下の手順で取得してください：

1. [Google AI Studio](https://makersuite.google.com/app/apikey) にアクセス
2. APIキーを生成
3. ノートブック実行時に環境変数 `GEMINI_API_KEY` として設定

### 2. 必要なライブラリ

ノートブック内で自動的にインストールされますが、主要な依存関係は以下の通りです：

```
google-genai
pillow
transformers
timm
accelerate
scikit-learn
matplotlib
seaborn
pandas
numpy
torch
```

### 3. ディレクトリ構造の準備

Google Colabで実行する場合、以下のディレクトリ構造をGoogle Driveに作成してください：

```
kujiramonaka_experiment/
├── base/
│   └── kujira_base.png  # ベース商品画像（必須）
├── images_formality/    # フォーマリティ軸の生成画像保存先
└── images_lifestage/    # ライフステージ軸の生成画像保存先
```

### 4. ベース画像の準備

`base/kujira_base.png` として、くじらもなかの商品画像を配置してください。この画像が生成の基準として使用されます。

## ファイル構造

```
kujiramonaka_experiment/
├── base/
│   └── kujira_base.png          # ベース商品画像（必須）
├── images_formality/            # フォーマリティ軸の生成画像
│   ├── formality_lv1_000.png
│   ├── formality_lv1_001.png
│   └── ...
├── images_lifestage/            # ライフステージ軸の生成画像
│   ├── lifestage_lv1_000.png
│   ├── lifestage_lv1_001.png
│   └── ...
├── formality_siglip_proj.npy    # フォーマリティ軸埋め込み（分析用）
├── formality_siglip_pool.npy    # フォーマリティ軸埋め込み（補助）
├── formality_labels.npy         # フォーマリティ軸ラベル
├── lifestage_siglip_proj.npy    # ライフステージ軸埋め込み（分析用）
├── lifestage_siglip_pool.npy    # ライフステージ軸埋め込み（補助）
└── lifestage_labels.npy         # ライフステージ軸ラベル
```

## クイックスタート

1. **Google Colabでノートブックを開く**
   - `banner_generation_experiment.ipynb` をGoogle Colabにアップロード

2. **セクション0を実行**
   - Google Driveをマウント
   - プロジェクトディレクトリを設定
   - GEMINI_API_KEYを設定

3. **セクション1を実行**
   - プロンプト定義を確認

4. **セクション2を実行（テスト生成）**
   ```python
   generate_formality_all_levels_once(n_images_per_level=1, preview_each_level=True)
   generate_lifestage_all_levels_once(n_images_per_level=1, preview_each_level=True)
   ```

5. **セクション3を実行**
   - 埋め込みベクトルを計算・保存

6. **セクション4を実行**
   - 分析結果を確認

7. **セクション5を実行**
   - 生成画像のギャラリーを表示

## 実行手順

### セクション0: 共通セットアップ

実験を再現する場合、画像データをGitHubから取得し、適宜ロードして分析するようお願いします。

現在はGoogle Driveから画像をロードするコードのみを記載しています。

1. Google Driveをマウント
2. プロジェクトルートとディレクトリを設定
   - `PROJECT_ROOT`: `/content/drive/MyDrive/kujiramonaka_experiment`
   - `BASE_IMAGE_DIR`: ベース商品画像を置く場所
   - `FORMALITY_DIR`: フォーマリティ軸の生成画像保存先
   - `LIFESTAGE_DIR`: ライフステージ軸の生成画像保存先
3. GEMINI_API_KEYを設定（`getpass`を使用して安全に入力）

### セクション1: プロンプト定義

各軸・各レベルのプロンプトを定義します。プロンプトは以下の要素で構成されます：

- **共通のプロダクト・画像制約** (`PRODUCT_IMAGE_CONSTRAINTS`)
  - くじらもなかの商品画像としての制約
  - 正方形（1:1）フォーマット
  - リアルな商品写真としての品質要件
  - 商品の形状やテクスチャを保持する制約

- **軸別のベースプロンプト**
  - `BASE_PROMPT_FORMALITY`: フォーマリティ軸のベースプロンプト
    - 贈答シーンのフォーマリティレベル（Level 1-5）の説明
    - 視覚的要素（背景、照明、小道具、空間配置、雰囲気）への変換指示
  - `BASE_PROMPT_LIFESTAGE`: ライフステージ軸のベースプロンプト
    - ライフステージペルソナレベル（Level 1-5）の説明
    - ライフスタイルと心理的傾向を視覚的選択に変換する指示

- **レベル別の指示文**
  - `FORMALITY_LEVEL_PROMPTS`: 各フォーマリティレベルの詳細な指示
  - `LIFESTAGE_LEVEL_PROMPTS`: 各ライフステージレベルの詳細な指示

**プロンプト構築関数：**
- `build_formality_prompt(level: int) -> str`: フォーマリティ軸のプロンプトを構築
- `build_lifestage_prompt(level: int) -> str`: ライフステージ軸のプロンプトを構築

**プロンプト設計の特徴：**
- ペルソナ情報のみが異なり、視覚的指示は明示的に含まれない
- AIモデルがペルソナ属性から視覚的差異を推論する設計

### セクション2: 画像生成クライアント & バッチ生成

Gemini 3 Pro Image Preview API（`gemini-3-pro-image-preview`）を使用して画像を生成します。

**主要な関数:**
- `generate_kujira_image()`: 個別画像生成関数
- `generate_formality_batch()`: フォーマリティ軸のバッチ生成
- `generate_lifestage_batch()`: ライフステージ軸のバッチ生成
- `generate_formality_all_levels_once()`: フォーマリティ軸の全レベル一括生成
- `generate_lifestage_all_levels_once()`: ライフステージ軸の全レベル一括生成

**テスト生成（各レベル1枚ずつ）:**
```python
generate_formality_all_levels_once(n_images_per_level=1, preview_each_level=True)
generate_lifestage_all_levels_once(n_images_per_level=1, preview_each_level=True)
```

**本番生成（各レベル10枚ずつ）:**
```python
generate_formality_all_levels_once(n_images_per_level=10, preview_each_level=True)
generate_lifestage_all_levels_once(n_images_per_level=10, preview_each_level=True)
```

**注意事項:**
- 各画像生成の間に2秒の待機時間が設定されています（APIレート制限対策）
- エラーが3回以上連続した場合、バッチ生成は停止します

### セクション3: 埋め込み取得・ラベル付け

#### セクション3-0: 依存関係（SigLIP2用）

必要なライブラリをインストール：
- `transformers`
- `timm`
- `accelerate`
- `torch`

#### セクション3-1: SigLIP2 モデル読み込み（定義）

SigLIP 2モデル（`google/siglip2-so400m-patch16-384`）を読み込みます。

#### セクション3-2: 画像パス/ラベル取得（定義）

生成された画像のパスとラベルを取得する関数を定義します。

#### セクション3-3: 埋め込み計算（定義・仕様固定）

画像を埋め込みベクトルに変換する関数を定義します。
- 正規化済み埋め込み（`proj`）: 分析用
- 生埋め込み（`pool`）: 補助データ

#### セクション3-4: 実行 & 保存

埋め込み計算を実行し、結果を保存します。

**出力ファイル:**
- `formality_siglip_proj.npy`: フォーマリティ軸の埋め込み（正規化済み、分析用）
- `formality_siglip_pool.npy`: フォーマリティ軸の埋め込み（生、補助）
- `formality_labels.npy`: フォーマリティ軸のラベル
- `lifestage_siglip_proj.npy`: ライフステージ軸の埋め込み（正規化済み、分析用）
- `lifestage_siglip_pool.npy`: ライフステージ軸の埋め込み（生、補助）
- `lifestage_labels.npy`: ライフステージ軸のラベル

### セクション4: 線形分類 & PCA可視化

#### セクション4-0: データ読み込み

保存された埋め込みデータとラベルを読み込みます。

#### セクション4-1: 指標計算（Repeated 5-fold + 距離指標を同梱）

以下の分析指標を計算します：

- **Accuracy**: Repeated 5-fold交差検証による分類精度
- **Macro F1**: マクロ平均F1スコア
- **Silhouette Score**: クラスタリングの品質（コサイン距離を使用）
- **Intra-class Diversity**: クラス内の多様性（平均距離）

#### セクション4-2: 可視化

以下の可視化を生成します：

- **PCA 2次元プロット**: 埋め込み空間の2次元への次元削減と可視化
- **混同行列**: 分類結果の詳細な評価

#### セクション4-3: 実行（サマリー表 + レベル別シルエット表 + 図）

分析結果を実行し、以下の出力を生成します：

- サマリー表: 全体の分析指標
- レベル別シルエットスコア表: 各レベルのクラスタリング品質
- 可視化図: PCAプロットと混同行列

### セクション5: 画像のギャラリー表示

生成された画像をレベルごとにグリッド表示します。各軸・各レベルの画像を整理して表示する機能を提供します。

## 出力ファイル

### 生成画像

生成された画像は以下の命名規則で保存されます：

- **フォーマリティ軸**: `images_formality/formality_lv{level}_{index:03d}.png`
  - 例: `formality_lv1_000.png`, `formality_lv1_001.png`, ..., `formality_lv5_009.png`
- **ライフステージ軸**: `images_lifestage/lifestage_lv{level}_{index:03d}.png`
  - 例: `lifestage_lv1_000.png`, `lifestage_lv1_001.png`, ..., `lifestage_lv5_009.png`

各レベル10枚生成する場合、合計100枚の画像が生成されます（5レベル × 2軸 × 10枚）。

### 埋め込みデータ

SigLIP 2モデルで計算された埋め込みベクトルとラベルが保存されます：

**フォーマリティ軸:**
- `formality_siglip_proj.npy`: 正規化済み埋め込みベクトル（分析用、主成分分析や分類に使用）
- `formality_siglip_pool.npy`: 生埋め込みベクトル（補助データ）
- `formality_labels.npy`: 各画像に対応するレベルラベル（1-5）

**ライフステージ軸:**
- `lifestage_siglip_proj.npy`: 正規化済み埋め込みベクトル（分析用、主成分分析や分類に使用）
- `lifestage_siglip_pool.npy`: 生埋め込みベクトル（補助データ）
- `lifestage_labels.npy`: 各画像に対応するレベルラベル（1-5）

### 分析結果

セクション4で生成される分析結果：

- **サマリー表**: 全体の分析指標（Accuracy, Macro F1, Silhouette Score, Intra-class Diversity）
- **レベル別シルエットスコア表**: 各レベルのクラスタリング品質
- **PCA 2次元プロット**: 埋め込み空間の可視化
- **混同行列**: 分類結果の詳細評価

## 注意事項

1. **APIレート制限**: Gemini APIにはレート制限があります。バッチ生成時は各画像生成の間に2秒の待機時間が設定されています。大量の画像を生成する場合は、時間がかかることを考慮してください。

2. **Google Colabのセッション**: 
   - SigLIP 2モデルの読み込みには時間がかかります。セッションがタイムアウトしないよう注意してください。
   - GPU推奨ですが、CPUでも動作します（処理速度が遅くなります）。

3. **ストレージ容量**: 
   - 生成画像は各レベル10枚×5レベル×2軸=100枚になります。Google Driveの容量を確認してください。
   - 埋め込みデータ（`.npy`ファイル）も保存されるため、十分な容量を確保してください。

4. **ベース画像**: 
   - ベース画像が正しく配置されていない場合、画像生成は失敗します。
   - `base/kujira_base.png` として、くじらもなかの商品画像を配置してください。

5. **実験の再現**: 
   - 実験を再現する場合、画像データをGitHubから取得し、適宜ロードして分析するようお願いします。
   - 現在のノートブックはGoogle Driveから画像をロードするコードのみを記載しています。

## トラブルシューティング

### APIキーエラー
- `GEMINI_API_KEY` が正しく設定されているか確認してください
- APIキーが有効期限内か確認してください

### 画像生成エラー
- ベース画像が正しいパスに配置されているか確認してください
- APIレート制限に達していないか確認してください

### 埋め込み計算エラー
- SigLIP 2モデルのダウンロードに失敗した場合、セッションを再起動してください
- GPUが利用可能か確認してください（CPUでも動作しますが遅いです）

## プロンプト設計の哲学

本実験のプロンプト設計では、以下の原則に基づいています：

1. **ペルソナ情報のみの差異**: 各レベル間で異なるのはペルソナ情報のみで、明示的な視覚的指示は含まれません
2. **内部解釈による視覚化**: AIモデルがペルソナ属性を内部で解釈し、適切な視覚的差異を推論します
3. **一貫性の確保**: 共通のプロダクト制約により、商品の本質的な特徴は維持されます

この設計により、ペルソナ情報がどのように視覚的表現に反映されるかを定量的に分析できます。

## 参考文献

- [Gemini API Documentation](https://ai.google.dev/docs)
- [SigLIP 2 Model](https://huggingface.co/google/siglip2-so400m-patch16-384)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

