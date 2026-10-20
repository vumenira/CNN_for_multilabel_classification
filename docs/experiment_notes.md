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

## Постобробка

Зараз у `notebooks/custom_cnn.ipynb`, експеримент 2: TTA hflip → pHash ≤ 2, λ = 0.5 → пороги на клас → ієрархія вгору.

1. TTA hflip: так / ні.
2. pHash: відстань ≤ 2 / 4 / 6; λ 0 / 0.25 / 0.5 / 0.75 / 1.
3. Пороги: спільний 0.5 / на клас; підбір на половині val, оцінка на іншій (у fold 0 лише 11 / 8 / 9 Moris / Motya / Biatrix).
4. Ієрархія: без / вгору (Moris, Motya → Cat; Biatrix → Dog) / вниз (Moris, Motya лише з Cat; Biatrix лише з Dog).
5. kNN по ембедингах CNN замість pHash.
6. Ridge поверх ймовірностей (потребує OOF).
