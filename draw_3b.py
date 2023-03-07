import sys
from strokes_gen_3b import *
# from gan_renderer import *
import torch
# import torch.optim as optim

def generate_strokes():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(25): #10 means there will produce 10 samples of the stroke
        f = torch.FloatTensor(1, 15).uniform_().to(device)
        
        # the first 12 num is shape
        stroke = f[:, :12].view(-1).cpu().numpy()
        
        stroke, data=draw3b(stroke, 64)
        # shape pass through the draw3b
        stroke = torch.from_numpy(stroke).to(device)
        stroke = stroke.view(64, 64, 1)
        
        # the last 3 num is color
        tempx = f[:, -3:].view(1, 1, 3)
        color_stroke = stroke * tempx
        # to data
        tempx=tempx.view(-1)*64
        data.append(tempx.tolist())

        canvas = color_stroke+1-stroke
        # print(canvas[0][0])
        # print(canvas[0][1])
        canvas = canvas.cpu().numpy()
        
        # cv2.imwrite('3b0_b/nblur100000/s' + str(step) + '.png', (a * 255).astype('uint8'))
        cv2.imwrite('./picture/picture' + str(i) + '.png', (canvas * 255).astype('uint8'))
        with open('./descript/picture' + str(i) + '.txt', 'w') as f:
            for item in data:
                for i in item:
                    f.write(str(i)+' ')
        
def test_strokes():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(25): #10 means there will produce 10 samples of the stroke
        data=[]
        with open('./test_descript/picture' + str(i) + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip("\n")
                line = line.split()
                for item in line:
                    data.append(float(item))
        data=np.array(data)
        stroke=redraw3b(data)
        tempx=torch.from_numpy(data[-3:]).view(1,1,3)
        tempx=tempx/64
        print(tempx)
        stroke = torch.from_numpy(stroke).view(64,64,1)
        color_stroke = stroke * tempx

        canvas = color_stroke+1-stroke
        # print(canvas[0][0])
        # print(canvas[0][1])
        canvas = canvas.cpu().numpy()
        
        # cv2.imwrite('3b0_b/nblur100000/s' + str(step) + '.png', (a * 255).astype('uint8'))
        cv2.imwrite('./test_picture/picture' + str(i) + '.png', (canvas * 255).astype('uint8'))




if __name__=="__main__":
    generate_strokes()
    test_strokes()