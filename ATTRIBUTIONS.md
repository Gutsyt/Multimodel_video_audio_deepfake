# Attributions and licences of third-party components

| Component | Use | Licence | Source |
|---|---|---|---|
| AASIST / AASIST-L code and pretrained weights (vendored in `third_party/aasist/`, unmodified) | Audio anti-spoofing inference | MIT, Copyright (c) 2021-present NAVER Corp. (see `third_party/aasist/LICENSE`, `NOTICE`) | https://github.com/clovaai/aasist, commit a04c9863 |
| Deep-Fake-Detector-v2-Model (downloaded at runtime, not redistributed) | Face-crop deepfake classification | Apache-2.0 per model card; training data undisclosed | https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model |
| google/vit-base-patch16-224-in21k (base of the above) | Architecture | Apache-2.0 | Dosovitskiy et al., ICLR 2021 |
| OpenCV and its Haar cascade | Frame decoding, face detection | Apache-2.0 (library); cascade file under the Intel/OpenCV licence in its header | https://opencv.org |
| PyTorch, Hugging Face Transformers, NumPy, FFmpeg | Runtime | Their respective open-source licences | – |
| scikit-image `astronaut` image | Test fixture only (face for software tests) | NASA image, public domain (per scikit-image documentation) | scikit-image |

Citations to include in the report: Jung et al. 2022 (AASIST, ICASSP); Dosovitskiy et al. 2021 (ViT); the ASVspoof 2019 database paper and DataShare record if used; Khalid et al. 2021 (FakeAVCeleb) only if used; Sanderson & Lovell 2009 and Korshunov & Marcel 2018 if VidTIMIT/DeepfakeTIMIT are used. Verify each reference against its original source before submission.
