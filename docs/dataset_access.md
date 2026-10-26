# Obtaining the data lawfully

Never commit dataset media. `data/` is gitignored. Read each dataset's terms yourself.

## FakeAVCeleb (primary)
1. Read the official README: https://github.com/DASH-Lab/FakeAVCeleb
2. Fill the Google request form linked in that README. After acceptance the authors send a download-script link. (I cannot do this for you, and I have not seen the form's terms.)
3. Download as the authors instruct into `data/raw/FakeAVCeleb/` (any location works; set `AVDD_DATA_ROOT`).
4. If the authors supply checksums, verify them. Otherwise record the archive sizes and sha256 of each archive in `docs/dataset_facts.md` (Phase 1) so the exact dataset version is identifiable.
5. Run, from the repository root with the venv active:
   ```bash
   python scripts/check_environment.py --json > env_report.json
   python scripts/inspect_dataset.py --root data/raw/FakeAVCeleb --out inventory_out --hash --workers 4
   ```
   `--hash` reads every media file once (slow on HDDs). Use `--max-probe 500` for a quick first look; sampled statistics are flagged as non-exhaustive in the output.
6. Send me `inventory_out/INVENTORY.md` (and `inventory.json` if possible). They contain relative file names, counts, and column value counts, no media. Check the dataset terms before sharing if they restrict sharing metadata.

## ASVspoof 2019 (supplementary, audio only; later phase)
Edinburgh DataShare record: https://datashare.ed.ac.uk/handle/10283/3336 (license shown there: Open Data Commons Attribution). LA.zip is 7.116 GB; PA is not needed. Cite the database paper (arXiv 1911.01601) and the DataShare record.

## Not needed yet
FaceForensics++ and AV-Deepfake1M: access terms unverified; nothing depends on them.
