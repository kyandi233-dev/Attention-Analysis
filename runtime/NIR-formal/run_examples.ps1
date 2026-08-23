$PythonExe = "python"

# 1. Check GPU, models and OpenCV trackers.
& $PythonExe .\run_pipeline.py check-env

# 2. Discover formal NIR videos under F:\正式实验 and E:\Data.
& $PythonExe .\run_pipeline.py discover

# 3. Recommended first smoke: 20 seconds, YOLO only, no RITnet.
& $PythonExe .\run_pipeline.py run --subject sub-056 --root "E:\Data" --duration-sec 20 --tracker none --skip-ritnet

# 4. One-minute integrated trial: YOLO every 10 frames + CSRT + RITnet on GPU.
& $PythonExe .\run_pipeline.py run --subject sub-056 --root "E:\Data" --duration-sec 60 --tracker csrt --redetect-interval 10

# 5. Direct video path is also supported.
& $PythonExe .\run_pipeline.py run --video "E:\Data\sub-056_\nir\sub-056_nir.avi" --duration-sec 60 --tracker csrt

# Full video requires an explicit safety flag. Do not use before short-video review.
# & $PythonExe .\run_pipeline.py run --subject sub-056 --root "E:\Data" --full-video --tracker csrt
