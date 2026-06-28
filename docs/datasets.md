# Dataset setup

This repository does not distribute dataset files, labels, challenge test inputs, or private manifests.

## MILK10k training data

Set the following paths in `configs/milk10k_reference.yaml`:

- `data.train_image_dir`: folder containing one subfolder per lesion;
- `data.ground_truth_csv`: ground-truth CSV containing a lesion identifier column and the eleven class columns.

The expected class order is:

```text
AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC
```

The manifest builder checks each lesion folder for supported image extensions (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`). It assigns the image with the lower parsed ISIC number as clinical and the higher parsed number as dermoscopic, matching the reference workflow.

## Test data

For inference, pass a test root containing one directory per lesion. The same image-pairing rule is used. The inference script writes a CSV containing a `lesion` column followed by raw stitched probabilities for the eleven classes.
