# くじらもなかバナー生成実験

このリポジトリには、「くじらもなか」という商品のプロモーションバナー画像を生成し、分析する実験のコードが含まれています。

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

## 実行手順

### セクション0: 共通セットアップ

1. Google Driveをマウント
2. プロジェクトルートとディレクトリを設定
3. GEMINI_API_KEYを設定

### セクション1: プロンプト定義

各軸・各レベルのプロンプトを定義します。プロンプトは以下の要素で構成されます：

- 共通のプロダクト・画像制約
- 軸別のベースプロンプト（フォーマリティ軸またはライフステージ軸）
- レベル別の指示文

### セクション2: 画像生成

Gemini 3 Pro Image APIを使用して画像を生成します。

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

### セクション3: 埋め込み取得

SigLIP 2モデル（`google/siglip2-so400m-patch16-384`）を使用して、生成画像の埋め込みベクトルを計算します。

出力ファイル：
- `formality_siglip_proj.npy`: フォーマリティ軸の埋め込み（正規化済み）
- `formality_siglip_pool.npy`: フォーマリティ軸の埋め込み（生）
- `formality_labels.npy`: フォーマリティ軸のラベル
- `lifestage_siglip_proj.npy`: ライフステージ軸の埋め込み（正規化済み）
- `lifestage_siglip_pool.npy`: ライフステージ軸の埋め込み（生）
- `lifestage_labels.npy`: ライフステージ軸のラベル

### セクション4: 分析

線形分類とPCA可視化により、生成画像の分析を行います。

**分析指標:**
- **Accuracy**: 5-fold交差検証による分類精度
- **Macro F1**: マクロ平均F1スコア
- **Silhouette Score**: クラスタリングの品質（コサイン距離）
- **Intra-class Diversity**: クラス内の多様性（平均距離）

**可視化:**
- PCA 2次元プロット
- 混同行列

### セクションX: ギャラリー表示

生成された画像をレベルごとにグリッド表示します。

## 出力ファイル

### 生成画像

- `images_formality/formality_lv{level}_{index:03d}.png`
- `images_lifestage/lifestage_lv{level}_{index:03d}.png`

### 埋め込みデータ

- `formality_siglip_proj.npy`: フォーマリティ軸の埋め込み（分析用）
- `formality_siglip_pool.npy`: フォーマリティ軸の埋め込み（補助）
- `formality_labels.npy`: フォーマリティ軸のラベル
- `lifestage_siglip_proj.npy`: ライフステージ軸の埋め込み（分析用）
- `lifestage_siglip_pool.npy`: ライフステージ軸の埋め込み（補助）
- `lifestage_labels.npy`: ライフステージ軸のラベル

## 注意事項

1. **APIレート制限**: Gemini APIにはレート制限があります。バッチ生成時は各画像生成の間に2秒の待機時間が設定されています。

2. **Google Colabのセッション**: SigLIP 2モデルの読み込みには時間がかかります。セッションがタイムアウトしないよう注意してください。

3. **ストレージ容量**: 生成画像は各レベル10枚×5レベル×2軸=100枚になります。Google Driveの容量を確認してください。

4. **ベース画像**: ベース画像が正しく配置されていない場合、画像生成は失敗します。

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

## 参考文献

- [Gemini API Documentation](https://ai.google.dev/docs)
- [SigLIP 2 Model](https://huggingface.co/google/siglip2-so400m-patch16-384)
- [Transformers Library](https://huggingface.co/docs/transformers)

