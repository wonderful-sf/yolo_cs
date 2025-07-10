import cv2 as cv
import numpy as np
from ultralytics import YOLO
import mss
import torch
import pyautogui
# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获取全屏分辨率
screen_w, screen_h = pyautogui.size()

# 只截取中心三分之一区域
crop_w = screen_w // 3
crop_h = screen_h // 3
x1 = (screen_w - crop_w) // 2
y1 = (screen_h - crop_h) // 2

# 模型输入尺寸
model_size = 640
model = YOLO("CS2.pt").to(device)

def to_tensor(img_np):
    # 1. 缩放到 model_size×model_size
    img_resized = cv.resize(img_np, (model_size, model_size))
    # 2. 转为 float32 并归一化到 [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    # 3. HWC 转为 CHW
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    # 4. 转为 Tensor 并添加 batch 维度
    return torch.from_numpy(img_transposed).unsqueeze(0).to(device)

cv.namedWindow('cs', cv.WINDOW_NORMAL)
cv.setWindowProperty('cs', cv.WND_PROP_TOPMOST, 1)
cv.resizeWindow('cs', crop_w , crop_h)

with mss.mss() as sct:
    monitor = {"top": y1, "left": x1, "width": crop_w, "height": crop_h}

    while True:
        try:
            # 使用 mss 截图
            screen = np.array(sct.grab(monitor))
            img = cv.cvtColor(screen, cv.COLOR_BGRA2BGR)
            # 3. 预处理并转换为 Tensor
            img_tensor = to_tensor(img)
            # 4. 推理并绘制检测框
            with torch.no_grad():
                output = model(img_tensor)
            result = output[0]
            plot_img = result.plot()
            cv.imshow('cs', plot_img)
            if cv.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print(f"错误：{e}")
            break

cv.destroyAllWindows()
