# Build and Run

Environment (Windows, conda)
1) Activate env in cmd (not PowerShell):
   "C:\\Users\\<you>\\anaconda3\\Scripts\\activate" pandaset
2) Build BEVFusion custom ops:
   python projects\\BEVFusion\\setup.py develop
   - Requires MSVC Build Tools and matching CUDA toolkit
3) Optional checks:
   - python check_transforms_registration.py
   - python test_registration.py

Data
- Generate infos with image+calibration fields:
  python tools\\create_pandaset_infos.py --root-dir data\\pandaset

Train
- From repo root in cmd:
  python tools\\train.py projects\\BEVFusion\\configs\\bevfusion_pandaset.py

Notes
- If loading from a pretrained checkpoint, place it locally and set `load_from`.
- On first run, Swin backbone weights download; ensure network access or provide local file.

