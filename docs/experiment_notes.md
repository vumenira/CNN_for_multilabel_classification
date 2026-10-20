# Експерименти

## Навчання

Базова CNN, 10 епох (run `86ea605c`): val macro F1 0.331; Cat 0.836, Dog 0.820, Moris / Motya / Biatrix 0 на всіх епохах при порозі 0.5. `val_loss` ще падав на 10-й епосі.

1. Більше епох: 10 / 20 / 40.
2. Поріг для Moris / Motya / Biatrix нижче 0.5: спільний / на клас (див. «Постобробка», п.3).
3. `pos_weight` у `BCEWithLogitsLoss` для рідкісних.

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
11. TTA з `HorizontalFlip`.

**Не робити:** `ToGray`, `HueSaturationValue`, `ColorJitter`, `VerticalFlip`, `Transpose`, `RandomRotate90`, `ElasticTransform`, `GridDistortion`, `RandomGridShuffle`.

## Постобробка

Зараз у `notebooks/custom_cnn.ipynb`, експеримент 2: TTA hflip → pHash ≤ 2, λ = 0.5 → пороги на клас → ієрархія вгору.

Run `86ea605c`, val macro F1 по шарах: модель 0.331 → TTA 0.332 → pHash 0.374 → пороги 0.547 → ієрархія 0.547.
Пороги: Cat 0.35, Dog 0.4, Moris 0.2, Motya 0.05, Biatrix 0.15.
Kaggle public: baseline 0.324, постобробка 0.445.

1. TTA hflip: так / ні.
2. pHash: відстань ≤ 2 / 4 / 6; λ 0 / 0.25 / 0.5 / 0.75 / 1.
3. Пороги: спільний 0.5 / на клас; підбір на половині val, оцінка на іншій (у fold 0 лише 11 / 8 / 9 Moris / Motya / Biatrix).
4. Ієрархія: без / вгору (Moris, Motya → Cat; Biatrix → Dog) / вниз (Moris, Motya лише з Cat; Biatrix лише з Dog).
5. kNN по ембедингах CNN замість pHash.
6. Ridge поверх ймовірностей (потребує OOF).

## Transfer learning

Архітектура: `docs/transfer_learning.md`.


1. ConvNeXt-B in22k (`convnext_base.fb_in22k_ft_in1k_384`), вхід 384.
2. Постобробка, як для CNN.
3. EffNet-B5 NS (`tf_efficientnet_b5.ns_jft_in1k`), вхід 456; ансамбль з п. 1.
4. ConvNeXt-B DINOv3 (`convnext_base.dinov3_lvd1689m`), вхід 384: архітектура як у п. 1, різниця лише в передтренуванні; кандидат в ансамбль.
5. Контроль: EffNet-B1 in1k (модель викладача), вхід 240.
7. Lr для backbone менший, ніж для голови; `drop_path_rate`.
8. kNN по ембедингах замість pHash.
9. Фінал: навчання на всіх 5 фолдах.

**Не робити:** зовнішні дані; ваги не з timm, крім ImageNet; ручне перемаркування;