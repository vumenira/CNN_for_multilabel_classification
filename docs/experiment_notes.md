# Експерименти

## Аугментації

1. Без аугментацій.
2. Поточний `train_transforms`.
3. `scale` у `RandomResizedCrop`: 0.8 / 0.9.
4. `p` у `OneOf`: 0 / 0.2 / 0.4.
5. `brightness_by_max`: True / False.
6. `CoarseDropout`.
7. CutMix, CMO.
8. `WeightedRandomSampler` для рідкісних.
9. Розмір входу: 224 / 240×288 / більший.
10. `Normalize`: 0.5 / статистики train.
11. `97098f` з міткою Moris і без неї.
12. TTA з `HorizontalFlip`.

**Не робити:** `ToGray`, `HueSaturationValue`, `ColorJitter`, `VerticalFlip`, `Transpose`, `RandomRotate90`, `ElasticTransform`, `GridDistortion`, `RandomGridShuffle`.

**Джерела:** лекція 7, §1–2; [Happywhale, 1 місце](https://github.com/knshnb/kaggle-happywhale-1st-place); [SIIM-ISIC Melanoma, 1 місце](https://arxiv.org/abs/2010.05351); [CMO](https://arxiv.org/abs/2112.00412); [LVIS](https://arxiv.org/abs/1908.03195); [Balestriero et al.](https://arxiv.org/abs/2204.03632).
