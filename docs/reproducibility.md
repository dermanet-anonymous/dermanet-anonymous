# Reproducibility

## Reference configuration

The default YAML configuration preserves the main source notebook's core setup:

- EfficientNetV2-XL backbone: `tf_efficientnetv2_xl.in21k_ft_in1k`
- feature levels: `(2, 3, 4)`
- embedding dimension: `512`
- fusion MLP: `1024 -> 512 -> 256`
- training image size: `576`
- batch size: `8`
- gradient accumulation: `8`
- AdamW learning rate: `9.4e-5`
- weight decay: `7.26e-5`
- cosine minimum learning-rate ratio: `0.01`
- label smoothing: `0.10`
- modality-dropout probability: `0.15`
- logit-adjustment coefficient: `0.12`

## Commands

Train from scratch:

```bash
python scripts/train_milk10k.py --config configs/milk10k_reference.yaml
```

Run paired-image inference:

```bash
python scripts/infer_milk10k.py \
  --config configs/milk10k_reference.yaml \
  --checkpoint checkpoints/best_milk10k.pth \
  --input-root /path/to/test_input \
  --output outputs/milk10k_probabilities.csv
```

## Evaluation behavior

Validation uses the identity image and a horizontal flip, averaging head logits before applying soft probability stitching. Test inference can use eight geometric views: four rotations (0, 90, 180, 270 degrees), each with and without a horizontal flip.

The implementation deliberately omits any threshold forcing, rare-class rescue, or leaderboard-specific post-processing so exported probabilities represent the model output itself.
