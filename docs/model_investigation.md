# Pretrained model investigation (24 Oct 2026)

Legend: [V] verified by me here; [V-2nd] reported by a third party; [U] unverified (needs your machine).

## Audio
| Candidate | License | Weights | Deps | Findings | Decision |
|---|---|---|---|---|---|
| **AASIST / AASIST-L** (clovaai/aasist, commit a04c986) | MIT (NAVER) [V: LICENSE read] | In the repo: 1.28 MB / 0.43 MB; sha256 in `third_party/aasist/SHA256SUMS.txt` [V] | torch, numpy only for inference [V: AASIST.py imports] | Loads with `strict=True`; 297,866 / 85,306 params (matches README for -L) [V]; deterministic CPU inference ~1-1.5 s per 4 s window on 1 core [V]. Upstream: EER 0.83% / 0.99% on ASVspoof 2019 LA eval (in-domain) [V-2nd: upstream README]. A third-party benchmark reports ~43-44% EER on in-the-wild audio, near chance [V-2nd]. Random noise scored as "spoof" [V]. | **Use (default AASIST).** Strong limitation on real-world audio is stated in every result. |
| W2V2-AASIST (SpeechAntiSpoofingBenchmarks HF org) | MIT [V-2nd] | LA_model.pth plus base XLS-R 300M checkpoint (large) | Wrapper code; fairseq-style checkpoint [U] | Same benchmark reports far better cross-dataset EER (e.g. 11.22% in-the-wild) [V-2nd] | **Not now.** Large weights and uncertain dependencies; revisit only if time remains. |
| Other HF wav2vec2 audio classifiers | Varied | Hub | transformers | Not investigated in depth; training data/licences unclear | Not used |

## Visual
| Candidate | License | Weights | Deps | Findings | Decision |
|---|---|---|---|---|---|
| **prithivMLmods/Deep-Fake-Detector-v2-Model** | Apache-2.0 [V-2nd: model card] | HF Hub, ViT-base, safetensors | torch, transformers, pillow | Image classifier; labels Realism/Deepfake [V-2nd]; training data "curated dataset", undisclosed; card warns of poor generalization and no video support [V-2nd]; vendor metrics not reproduced. **Not downloaded or run by me: huggingface.co is unreachable from my sandbox.** | **Use, pending your probe (A3).** Applied per face crop; frame scores aggregated by mean. |
| Arko007/deepfake-detector-dfd-sota (CLIP-ViT-L, 12-frame) | Apache-2.0 [V-2nd] | 1.28 GB | custom architecture needed | Its own card reports ~47-55% real-class detection and a tiny real set | Rejected: heavy, custom code, weak reported results |
| Classic video detectors (DFDC winners, DeepfakeBench weights) | Mixed; training data (DFDC etc.) carries its own terms | Large / Google Drive | Old PyTorch stacks | Not investigated further: licence and dependency risk for a 3-day window | Not used |

## Face detection
OpenCV Haar frontal-face cascade bundled with opencv-python-headless 4.x [V: ran on a public-domain test photo; cascade header carries the Intel/OpenCV licence]. YuNet (MIT, opencv_zoo) is a better detector but its ONNX file is in Git LFS (official sha256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` read from the LFS pointer) and its host was unreachable from my sandbox, so it is deferred. OpenCV 5.x needs a different YuNet file, so `opencv-python-headless<5` is pinned.

## Why this combination is only a demo-grade detector
The two models were trained on unrelated data, none of it FakeAVCeleb-style audio-visual forgeries, and neither is validated here. Expect false alarms and misses. The scientific contribution of the prototype is the *pipeline*: preprocessing, per-branch evidence, explicit modality-status handling, documented fusion, and transparent limitation reporting.
