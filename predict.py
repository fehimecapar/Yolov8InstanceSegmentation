from ultralytics import YOLO

model = YOLO('yolov8s-seg-custom.pt')

model.predict(source = 'test/images/t_a1.jpg', show = True, save = True, show_labels = True, show_conf=True, conf = 0.5, save_txt = False, save_crop = False, line_width = 2)