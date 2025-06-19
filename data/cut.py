# coding=utf-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知和警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用gpu0

# from train_def import *
from global_annos import *
from global_ import *
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.utils.data

BATCH_SIZE = 1
EPOCH = 1


class cutDataset(Dataset):

    def __init__(self, data_path, label_path):  ###  transform 我没写
        self.data = self.get_img_label(data_path)  ## 图的位置列表
        self.label = self.get_img_label(label_path)  ## 标签的位置列表

        self.annos_img = self.get_annos_label(self.data)  # 图的位置列表 输入进去  吐出  结节附近的图的【【图片位置，结节中心，半径】列表】
        self.annos_label = self.get_annos_label(self.label)  # 112

    def __getitem__(self, index):
        img_all = self.annos_img[index]
        label_all = self.annos_label[index]
        img = np.load(img_all[0])  # 载入的是图片地址
        label = np.load(label_all[0])  # 载入的是label地址
        cut_list = []  ##  切割需要用的数

                for i in range(len(img.shape)):   ###  0,1,2   →  z,y,x
            if i == 0:
                a = img_all[1][-i - 1] - 8  ### z
                b = img_all[1][-i - 1] + 8
            else:
                a = img_all[1][-i-1]-48   ### z
                b = img_all[1][-i-1]+48   ###
            if a<0:
                if i == 0:
                    a = 0
                    b = 96
                else:
                    a = 0
                    b = 96
            elif b>img.shape[i]:
                if i == 0 :
                    a = img.shape[i] - 16
                    b = img.shape[i]
                else:
                    a = img.shape[i]-96
                    b = img.shape[i]
            else:
                pass

            cut_list.append(a)
            cut_list.append(b)

        cut_list = [round(i) for i in cut_list]
        img = img[cut_list[0]:cut_list[1], cut_list[2]:cut_list[3], cut_list[4]:cut_list[5]]  ###  z,y,x
        label = label[cut_list[0]:cut_list[1], cut_list[2]:cut_list[3], cut_list[4]:cut_list[5]]  ###  z,y,x
        one_path_img = str(Path(output_path) / "bbox_image_npy" / Path(img_all[0]).parent.name / (
                Path(img_all[0]).stem + f'_{img_all[-1]}.npy'))
        Path(one_path_img).parent.mkdir(exist_ok=True, parents=True)
        np.save(one_path_img, img)
        one_path_label = str(Path(output_path) / "bbox_mask_npy" / Path(img_all[0]).parent.name / (
                Path(img_all[0]).stem + f'_{img_all[-1]}.npy'))
        Path(one_path_label).parent.mkdir(exist_ok=True, parents=True)
        np.save(one_path_label, label)
        one_list = [str(one_path_img), str(one_path_label), str(img_all[1])]

        # img = np.expand_dims(img,0)  ##(1, 96, 96, 96)
        # img = torch.tensor(img)
        # img = img.type(torch.FloatTensor)
        # label = torch.Tensor(label).long()  ##(96, 96, 96) label不用升通道维度
        # torch.cuda.empty_cache()
        return one_list  ### 从这里出去还是96*96*96

    def __len__(self):
        return len(self.annos_img)

    @staticmethod
    def get_img_label(data_path):  ###  list 地址下所有图片的绝对地址

        img_path = []
        for t in data_path:  ###  打开subset0，打开subset1
            data_img_list = os.listdir(t)  ## 列出图
            img_path += [os.path.join(t, j) for j in
                         data_img_list]  ##'/public/home/menjingru/dataset/sk_output/bbox_image/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.npy'
        img_path.sort()
        return img_path  ##返回的也就是图像路径 或 标签路径

    @staticmethod
    def get_annos_label(img_path):
        annos_path = []  # 这里边要装图的地址，结节的中心，结节的半径    要小于96/4 # ###半径最大才12

        ### ok   ,   anoos 是处理好的列表了，我只需要把他们对比一下是否在列表里，然后根据列表里的坐标输出一个列表  就可以了   在__getitem__里边把它切下来就行

        for u in img_path:  # 图的路径
            name = Path(u).stem
            for one in annos_list:  # 遍历有结节的图
                if one[0] == name:  # 如果有结节的图的名字 == 输入的图的名字
                    for l in range(len(one[1])):  # 数一数有几个结节
                        annos_path.append(
                            [u, [one[1][l][0], one[1][l][1], one[1][l][2]], one[1][l][3], l])  # 图的地址，结节的中心
        return annos_path  # ###半径最大才12


torch.cuda.empty_cache()  # 时不时清下内存

data_path = []  # 装图所在subset的绝对地址，如 [D:\datasets\sk_output\bbox_image\subset0,D:\datasets\sk_output\bbox_image\subset1,..]
label_path = []  # 装标签所在subset的绝对地址，与上一行一致，为对应关系
for i in range(0, 10):  # 0,1,2,3,4,5,6,7   训练集
    data_path.append(str(Path(bbox_img_path) / f'subset{i}'))  # 放入对应的训练集subset的绝对地址
    label_path.append(str(Path(bbox_msk_path) / f'subset{i}'))
dataset_train = cutDataset(data_path, label_path)  # 送入dataset
print(len(dataset_train))
train_loader = torch.utils.data.DataLoader(dataset_train,  # 生成dataloader
                                           batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=0)  # 16)  # 警告页面文件太小时可改为0
print("train_dataloader_ok")

all_msg_list = []
for epoch in range(1, EPOCH + 1):  # 每一个epoch  训练一轮   检测一轮
    tqdr = tqdm(enumerate(train_loader))  # 用一下tqdm函数，也就是进度条工具（枚举）

    for batch_index, one_list in tqdr:
        all_msg_list.append([i[0] for i in one_list])
df = pd.DataFrame(all_msg_list, columns=['img_path', 'lbl_path', 'msg'])  # msg是结节的中心 z,y,x
df.to_excel(msg_path)

