# adaptive-semantic-similarity [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oAx3skn0PCLnWOWBpY33Yns3Ql7nQjop?usp=sharing)

Adaptive Semantic Similarity 是一個高效、靈活的文本相似度計算工具，專門設計用於處理各種長度的文本，即使超出預訓練模型的最大token限制。

## 主要特點

1. **基於 Token 的文本分割**: 
   - 使用語言模型的 token 進行精確的文本分割，而非傳統的基於字符長度的方法。
   - 能夠處理超過預訓練模型最大 token 限制（如 512）的長文本。

2. **兩種高效算法**:
   - `AggregateEmbeddings`: 通過聚合分塊文本的嵌入來計算相似度。
   - `SlidingWindow`: 使用滑動窗口技術，支持 mean 和 max 兩種聚合方法。

3. **靈活的模型選擇**:
   - 默認使用 'sentence-transformers/all-mpnet-base-v2' 模型。
   - 支持用戶根據需求使用其他兼容的預訓練模型。

4. **出色的長文本處理能力**:
   - 即使在處理遠超模型最大 token 限制的文本時，仍能保持高精度。

5. **適應不同應用場景**:
   - `AggregateEmbeddings`: 適用於大多數一般情況，提供穩定可靠的相似度計算。
   - `SlidingWindow (max)`: 特別適合部分抄襲檢測，能夠捕捉局部高度相似的片段。
   - `SlidingWindow (mean)`: 在長文本比較中表現出與人類直觀判斷相符的結果。

## 工作原理

1. **文本向量化**：
   - 使用預訓練的語言模型（默認為 'sentence-transformers/all-mpnet-base-v2'）將輸入文本轉換為高維向量表示。
   - 這些向量捕捉了文本的語義信息，使得語義相似的文本在向量空間中距離較近。

2. **基於 Token 的文本分割**：
   - 對於超過模型最大 token 限制的長文本，我們使用基於 token 的分割方法，而非傳統的字符串長度分割。
   - 這確保了分割後的文本片段保持語義完整性。

3. **相似度計算**：
   - 使用餘弦相似度來衡量向量間的相似程度。餘弦相似度值範圍從 -1 到 1，其中 1 表示完全相似。
   - 對於長文本，我們提供了不同的聚合策略來計算整體相似度。

4. **聚合策略**：
   - `AggregateEmbeddings`：將分割後的文本片段的嵌入向量聚合，然後計算相似度。
   - `SlidingWindow`：使用滑動窗口方法，支持 mean 和 max 兩種聚合方式，分別適用於不同場景。

## 性能數據

以下是在不同長度文本上的性能測試結果（測試環境：RTX 3060 Laptop）：

| 文本長度 (Token) | AE | SW (mean) | SW (max) |
|----------------|---------|---------|---------|
| 136 vs 135     | 0.1600s | 0.0340s | 0.0290s |
| 1342 vs 1332   | 0.3520s | 0.3310s | 0.3379s |
| 13402 vs 13302 | 1.8357s | 3.9340s | 3.8526s |

注：AE = Aggregate Embeddings, SW = Sliding Window

查看[完整測試結果](https://github.com/yesaouo/adaptive-semantic-similarity/blob/main/test.txt)以獲取更詳細的性能數據。

## 算法特點總結

- **AggregateEmbeddings**: 在大多數情況下表現穩定，提供可靠的相似度計算。
- **SlidingWindow (max)**: 對於檢測部分抄襲或高度相似的文本片段特別有效。
- **SlidingWindow (mean)**: 在比較長文本時，結果更符合人類的直觀判斷。

## 應用場景

- 文本相似度分析
- 抄襲檢測系統
- 長文本比較
- 智能文章推薦引擎

## 技術要求

- PyTorch
- Transformers
- NumPy

## 快速開始

```python
from adaptive_semantic_similarity import AggregateEmbeddings, SlidingWindow

# 使用 AggregateEmbeddings
ae = AggregateEmbeddings()
similarity_ae = ae.get_similarity("Text1...", "Text2...")

# 使用 SlidingWindow
sw = SlidingWindow()
similarity_sw_mean = sw.get_similarity("Text1...", "Text2...", method="mean")
similarity_sw_max = sw.get_similarity("Text1...", "Text2...", method="max")

print(f"AggregateEmbeddings 相似度: {similarity_ae}")
print(f"SlidingWindow (mean) 相似度: {similarity_sw_mean}")
print(f"SlidingWindow (max) 相似度: {similarity_sw_max}")
```

## 自定義模型

您可以輕鬆地使用自定義預訓練模型：

```python
#預設 model_name='sentence-transformers/all-mpnet-base-v2'
custom_ae = AggregateEmbeddings(model_name='your-custom-model-name')
custom_sw = SlidingWindow(model_name='your-custom-model-name')
```

## 自定義滑動窗口的步幅

您可以輕鬆地調整滑動窗口的步幅：

```python
#stride = (chunk_size - 2) // N
custom_sw = SlidingWindow(N=4)
```

## 注意事項

- 默認模型主要支持英文。如需處理其他語言，請選擇適合的預訓練模型。
- 處理時間可能因硬件設備和文本長度而異。
- 本工具能夠有效處理超過預訓練模型最大 token 限制的文本，特別適合長文本比較。

開始使用前，請克隆倉庫並安裝所需的依賴項！我們歡迎貢獻和反饋。
