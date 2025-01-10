from ultralytics import ASSETS, SAM

# Load a model
model = SAM("sam2.1_t.pt")

# Display model information (optional)
model.info()

# Segment image or video
results = model('bus.jpg') # 图片推理
#results = model('d.jpg') # 视频推理    

# Display results
for result in results:
    result.show()
