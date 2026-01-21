# TODO List

## ~~Checkpoint 保存 config~~ ✅ 已完成

已在 [src/pretrain/trainer.py:80](src/pretrain/trainer.py#L80) 实现。

---

## ~~Next Visit Prediction 任务~~ ✅ 已完成

已添加第 6 个下游任务：**Next Visit Prediction**

### 任务描述
- 预测患者下次住院会出现哪些 token（诊断、手术、药物、检验）
- 输入：患者历史 admission 的所有 token（不含目标住院）
- 输出：下次 admission 的 token 集合
- 评估指标：Recall@K, NDCG@K

### 实现内容

- [x] 1. 在 `DownstreamTask` 枚举中添加 `NEXT_VISIT = "next_visit"` - [data_utils.py:21](src/finetune/data_utils.py#L21)
- [x] 2. 在 `TASK_CONFIGS` 中添加配置 - [data_utils.py:31](src/finetune/data_utils.py#L31)
- [x] 3. 创建 `NextVisitDataset` - [data_utils.py:305-519](src/finetune/data_utils.py#L305-L519)
- [x] 4. 创建 `HATForNextVisit` 模型 - [model.py:279-523](src/finetune/model.py#L279-L523)
- [x] 5. 创建 `NextVisitTrainer` 使用 Recall@K 和 NDCG@K 评估 - [trainer.py:579-1039](src/finetune/trainer.py#L579-L1039)
- [x] 6. 更新 `run_finetune.py` 支持 next_visit 任务 - [run_finetune.py:281-411](run_finetune.py#L281-L411)

### 使用方法

```bash
# 单独运行 next_visit 任务
python run_finetune.py --task next_visit --pretrained checkpoints/best.pt

# 自定义 K 值
python run_finetune.py --task next_visit --pretrained checkpoints/best.pt --k-values 10,20,50

# 运行所有任务（包括 next_visit）
python run_finetune.py --all --pretrained checkpoints/best.pt
```

### 与其他任务的区别

| 任务 | 输出类型 | 评估指标 |
|------|----------|----------|
| mortality, readmission, prolonged_los | 二分类 | AUROC, AUPRC |
| icd_chapter | 多分类 | AUROC, AUPRC |
| icd_category_multilabel | 多标签分类 | AUROC, AUPRC |
| **next_visit** | **集合预测** | **Recall@K, NDCG@K** |
