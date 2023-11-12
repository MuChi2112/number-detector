from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter


class Paint(object):

    def __init__(self):
        self.root = Tk()
        self.root.title('Verify Number By AI')

        # Creating Canvas
        self.c = Canvas(self.root, bg='white', width=280, height=280)
        self.image1 = Image.new('RGB', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.image1)
        self.c.grid(row=1, columnspan=6)

        # Create "Verify" button
        self.classify_button = Button(self.root, text='Verify', command=lambda: self.classify(self.c))
        self.classify_button.grid(row=0, column=0, columnspan=2, sticky='EWNS')

        # Create "Clear" button
        self.clear = Button(self.root, text='Clear', command=self.clear)
        self.clear.grid(row=0, column=2, columnspan=2, sticky='EWNS')

        # Create "Save" button
        self.savefile = Button(self.root, text='Save', command=self.savefile)
        self.savefile.grid(row=0, column=4, columnspan=2, sticky='EWNS')

        # Create prediction text box
        self.prediction_text = Text(self.root, height=2, width=10)
        self.prediction_text.grid(row=2, column=4, columnspan=2)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        self.color = 'black'
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line((self.old_x, self.old_y, event.x, event.y),
                           fill='black', width=self.line_width)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.c.delete("all")
        self.image1 = Image.new('RGB', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.image1)
        self.prediction_text.delete("1.0", END)

    def savefile(self):
        f = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("png file", ".png")])
        if f is None:
            return
        self.image1.save(f)

    def classify(self, widget):
        img = self.image1.resize((28, 28), ImageGrab.Image.ANTIALIAS).convert('L')

        img = np.array(img)

        # 反色
        img = 255 - img

        # 二值化
        threshold = 128
        img = (img > threshold) * 255
        
        # 輕微模糊
        img = gaussian_filter(img, sigma=0.7)

        # 保存處理後的圖像
        img2 = Image.fromarray(img.astype(np.uint8))
        # print(img)
        # img2.save('classify_img.jpg')

        # 正規化
        img = img / 255

        # 轉為PyTorch張量並進行預測
        img = np.reshape(img, (1, 1, 28, 28))
        data = torch.FloatTensor(img).to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        self.prediction_text.delete("1.0", END)
        self.prediction_text.insert(END, predicted.item())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 载入模型
def loadModel():
    model = torch.load('cnn_augmentation_model.pt').to(device)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")

    # 载入既有的模型
    print('load model ...')
    model = loadModel()

    # 显示视窗
    Paint()