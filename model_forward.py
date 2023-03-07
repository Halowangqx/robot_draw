from tool_luhr import *


if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
output_num=63
learning_rate=1e-3



class strokeDataset(torch.utils.data.Dataset):
	def __init__(self, root, transforms):
		self.root = root
		self.transform = transforms
		# load all image files, sorting them to
		# ensure that they are aligned
		self.imgs = list(os.listdir(os.path.join(root, "picture")))

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, "picture", self.imgs[idx])
		img = Image.open(img_path).convert("RGB")
		if self.transform is not None:  # 对图片进行二次处理
			img = self.transform(img)
		# label_file=open(label_path,'r')
		# label_list=[]
		# for line in label_file:
		# 	data_line=line.strip("\n").split()
		# 	for i in data_line:
		# 		label_list.append(float(i))
		# label=torch.tensor(np.array(label_list))
		return img



def generate_ans(batchsize):
    std_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
            transforms.ToTensor(),
            std_normalize])
        

    dataset=strokeDataset("./",transforms=trans)
    dataloader=torch.utils.data.DataLoader(dataset,batchsize)
    model = torch.load('model_15_all_5.pth')
    model.eval()
    model.to(device)
    for input_tensor in dataloader:
        pred=model(input_tensor.to(device))
        print(pred.shape)
        break
    
generate_ans(10)