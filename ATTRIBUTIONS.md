# Attributions and licences

| Component | Use | Licence | Source |
|---|---|---|---|
| CLIP ViT-L/14 (downloaded at runtime) | frozen face-feature backbone | MIT | https://huggingface.co/openai/clip-vit-large-patch14 · Radford et al., ICML 2021 |
| wav2vec 2.0 XLS-R 300M (downloaded at runtime) | frozen speech-feature backbone | Apache-2.0 | https://huggingface.co/facebook/wav2vec2-xls-r-300m · Babu et al., 2021 |
| YuNet face detector (`src/avdd/assets/face_detection_yunet_2023mar.onnx`) | face boxes | MIT | https://huggingface.co/opencv/face_detection_yunet · Wu et al., 2023 |
| Trained heads (`src/avdd/assets/*_head.npz`) | face / voice classifiers | trained in this project | `scripts/train_heads.py` |
| FaceForensics++ (c23) | training / test | FaceForensics terms of use (research) | Rössler et al., ICCV 2019 |
| Celeb-DF / FakeAVCeleb frames | training / test | dataset terms (research) | Li et al., CVPR 2020; Khalid et al., NeurIPS 2021 |
| WildDeepfake | training / test | Apache-2.0 (HF release) | Zi et al., ACM MM 2020 |
| ASVspoof 2019 LA, ASVspoof 5 | training / test | ODC-By (HF release) | Wang et al., 2020; Wang et al., 2024 |
| In-the-Wild | training / test | Apache-2.0 (HF release) | Müller et al., Interspeech 2022 |
| PyTorch, Transformers, scikit-learn, OpenCV, FFmpeg, React, Motion | runtime / training | their open-source licences | – |
| scikit-image `astronaut` image | software-test fixture only | public domain | scikit-image |

Method references: Ojha et al., CVPR 2023 (UniversalFakeDetect); Khan & Dang-Nguyen, 2024 (CLIPping the Deception); Pascu et al., Interspeech 2024.
Dataset terms restrict use to research; check them before any commercial use.
