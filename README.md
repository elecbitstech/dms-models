# dms-models

Model artefacts for the i.MX93 Driver Monitoring System fleet.
Contents mirror `/opt/dms/downloads/` on reference device **C-00006**, pulled 2026-08-18.

Raw URL form:

```
https://raw.githubusercontent.com/elecbitstech/dms-models/main/<filename>
```

## Why the hashes matter

`downloads.json` on each device pins a **sha1 per file**, and `launcher.py` verifies it
at boot. On mismatch it **deletes the local file** and re-downloads from the URL above.
If the pinned sha1 and the file served here disagree, the board deletes its model, fails
to replace it, and never starts inference. Update the file and `downloads.json` together,
and update `downloads.json` **first**.

The Yocto recipe `dms-models.bb` additionally pins **sha256** for build-time fetching.

## Source models

| file | bytes | sha1 | sha256 (first 16) | model |
|---|---|---|---|---|
| `eye_state_int8.tflite` | 16880 | `6136bca333651f2df493bb72d129316b2aca0fc5` | `5cbaa188cba3de3d` | IR eye-state classifier |
| `face_detection_ptq.tflite` | 131248 | `8436d74058d81cfeb63986fbc835274907a028ea` | `b5bad49c265c451c` | NXP / MediaPipe BlazeFace, int8 |
| `face_landmark_ptq.tflite` | 653960 | `e5e2ebd15860a48de9fc9063a1819e20c6d82fd6` | `7d12ff9cacaa5059` | NXP / MediaPipe FaceMesh 468-pt |
| `facenet_int_quantized.tflite` | 23848680 | `a64cc9df1293b8e7efce03402691a4123350df5a` | `6ee9fcae748b26b8` | third-party FaceNet |
| `iris_landmark_ptq.tflite` | 735048 | `dfb3e5dbad7693399bebb42e76ac40ae554854f5` | `c3b187caa1f8fca2` | NXP / MediaPipe iris |
| `seatbelt_cls_final_int8.tflite` | 70504 | `8dbe36eb05a899b3bb0ee7fa1094be561978a2bb` | `62429b97547dc5b0` | belt/no-belt torso-crop CNN |
| `seatbelt_detection.tflite` | 3233113 | `2cdfdfa53b404e5028740a02b889018aee883d9b` | `1edc8fa4bed7eebe` | seatbelt detector |
| `yolov4_tiny_smk_call.tflite` | 6009728 | `cc30ee8eac666250e5f1e49bd2c76731a244cb29` | `49d1c41e5c389c3b` | phone + cigarette, DAY (day_v3) |
| `yolov4_tiny_smk_call_ir.tflite` | 6009520 | `a0567623eb038c05bbb19c9cae78c52ab12b516a` | `4e2af3397fb5caec` | phone + cigarette, IR/NIGHT |

## Pre-compiled NPU builds (Ethos-U65-256)

Produced by `vela --accelerator-config ethos-u65-256 --optimise Performance`.
Included because some deployed launcher revisions do not auto-compile; without them
a board silently falls back to CPU. Verify one is genuine with `grep -ac ethos-u <file>` (expect `1`).

| file | bytes | sha1 |
|---|---|---|
| `face_detection_ptq_vela.tflite` | 181216 | `e42ee64358bc2230e7ecdec1e193e696234e1cc9` |
| `face_landmark_ptq_vela.tflite` | 672800 | `ec808074714bfbf68a387fbdbf0cf52c80ba9ec6` |
| `facenet_int_quantized_vela.tflite` | 21213744 | `6d7dd5cd6e08d7faf7379e939769653c836b7417` |
| `iris_landmark_ptq_vela.tflite` | 713920 | `1c9f946d31c48167d502465b51bad85c967af0f9` |
| `seatbelt_cls_final_int8_vela.tflite` | 68368 | `df6aa4cebcbda3c74edaf8264fa3aee18eb23a96` |
| `seatbelt_detection_vela.tflite` | 2818192 | `bf7dd418396fea8ffcc021c1d2a5d5682057e226` |
| `yolov4_tiny_smk_call_ir_vela.tflite` | 5199440 | `4edbffb70d7b03a2ada5572ba7c31f074e507fd7` |
| `yolov4_tiny_smk_call_vela.tflite` | 5225312 | `b3bebfd8a97eb79f5d9b7456763870ec76a7de88` |

## day_v3 performance

`yolov4_tiny_smk_call.tflite`, measured on 1,074 in-car frames held out of training,
score >= 0.55, IoU >= 0.5, int8 as deployed:

| metric | previous day model | day_v3 |
|---|---|---|
| phone F1 | 0.349 | **0.887** |
| cigarette F1 | 0.063 | **0.865** |
| overall F1 | 0.237 | **0.879** |
| false-fire on empty-hand frames | 21.5% | **1.1%** |

Trained on 6,827 manifest lines from 5,369 unique frames, 6 drivers, 2 vehicles.
Anchors were refitted by k-means (IoU distance) for this data and **must match the runtime**:

```python
ANCHORS_DAY_V2 = [23, 21, 52, 15, 45, 33, 34, 58, 56, 73, 104, 67]
```

The runtime decodes `w = exp(raw) * anchor`, so a stale anchor list yields
confidently-scored boxes of the wrong size — which reads as an accuracy problem,
not a configuration one.

