# Dataset Notes

This release does not include dataset images, masks, split files, or archives.

Expected normalized layout:

```text
data/<dataset_name>/
|-- images/
|-- masks/
`-- splits/
    |-- train.txt
    |-- val.txt
    `-- test.txt
```

Images and masks are matched by relative stem. For example, `images/A/0001.jpg`
matches `masks/A/0001.png`, and a split entry can be either `A/0001` or
`A/0001.jpg`. Masks are converted to binary labels with `mask > 0`.

Normal samples should have empty binary masks. If an original dataset stores
normal images without mask files, create all-zero masks before training.

