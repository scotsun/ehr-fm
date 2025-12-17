# TODO List

## Checkpoint 保存 config

在 `src/pretrain/trainer.py` 的 `save_checkpoint` 方法中添加 config 保存，避免加载时需要从 state_dict 推断配置。

```python
# 在 checkpoint dict 中添加:
'config': model.config.to_dict(),
```

位置：[src/pretrain/trainer.py:72](src/pretrain/trainer.py#L72)

---

## Next Visit Prediction 任务

需要添加第 6 个下游任务：**Next Visit Prediction**

### 任务描述
- 预测患者下次住院会出现哪些 token（诊断、手术、药物、检验）
- 输入：患者历史 admission 的所有 token
- 输出：下次 admission 的 token 集合
- 评估指标：Recall@K, NDCG@K（已在 `src/baselines/metric.py` 中实现）

### 实现步骤

- [ ] 1. 在 `DownstreamTask` 枚举中添加 `NEXT_VISIT = "next_visit"`
- [ ] 2. 在 `TASK_CONFIGS` 中添加配置（prediction_time: "discharge", include_current: True）
- [ ] 3. 创建 `NextVisitDataset` 或修改 `FinetuneDataset` 支持集合预测
- [ ] 4. 创建 `NextVisitTrainer` 使用 Recall@K 和 NDCG@K 评估
- [ ] 5. 在 `create_downstream_labels.py` 中生成 next_visit 标签
- [ ] 6. 更新 `run_finetune.py` 支持 next_visit 任务

### 与其他任务的区别

| 任务 | 输出类型 | 评估指标 |
|------|----------|----------|
| mortality, readmission, prolonged_los | 二分类 | AUROC, AUPRC |
| icd_chapter | 多分类 | AUROC, AUPRC |
| abnormal_lab | 二分类 | AUROC, AUPRC |
| **next_visit** | **集合预测** | **Recall@K, NDCG@K** |

### 参考
- 队友实现：`master` 分支的 `src/metric.py`
- 已复制到：`src/baselines/metric.py`
