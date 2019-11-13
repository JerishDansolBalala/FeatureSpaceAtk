import sys
from PIL import Image
import os

def save_lst(lst, path):
    images = list(map(Image.open, lst))
    
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save(path)

decoder_level = ["imagenet_shallowest","imagenet_shallow","imagenet"]
attack_level = ["self_aug","selfaug_mean","selfaug_sigma","polygon","pgd"]

def file_addr(decoder,attack,i,j):
    return os.path.join(decoder,attack,str(i),"_%d_1.jpg"%j)

Good_Example =[(15,3),(17,3),(17,7), (11,3), (4,1), (4,0),(2,1), (14,3), (14,6), (14,5) , (6,0), (12,4)]

mode=1
if mode==0:
    task_dir = "model_depth"
    for i in range(1,20):
        for j in range(8):
            attack = "selfaugall"
            f_lst=[]
            f_lst.append(file_addr("imagenet","nat",i,j))
            for decoder in decoder_level:
                if os.path.exists(file_addr(decoder,attack,i,j)):
                    f_lst.append(file_addr(decoder,attack,i,j))
            os.makedirs(os.path.join(task_dir,"%d"%i),exist_ok=True)
            path = os.path.join(task_dir,"%d"%i,"%d.jpg"%j)
            save_lst(f_lst,path)

    task_dir = "attack"
    for i in range(1,20):
        for j in range(8):
            decoder = "imagenet"
            f_lst=[]
            f_lst.append(file_addr("imagenet","nat",i,j))
            for attack in attack_level:
                if os.path.exists(file_addr(decoder,attack,i,j)):        
                    f_lst.append(file_addr(decoder,attack,i,j))
            os.makedirs(os.path.join(task_dir,"%d"%i),exist_ok=True)
            path = os.path.join(task_dir,"%d"%i,"%d.jpg"%j)
            save_lst(f_lst,path)        
elif mode==1:
    task_dir = "demon\\model_depth"
    cnt=0
    for idx,(i,j) in enumerate(Good_Example):
        attack = "selfaugall"
        f_lst=[]
        f_lst.append(file_addr("imagenet","nat",i,j))
        for decoder in decoder_level:
            if attack == "selfaugall":
                if os.path.exists(os.path.join("add",file_addr(decoder,attack,idx //8 ,idx%8))):        
                    f_lst.append(os.path.join("add",file_addr(decoder,attack,idx //8 ,idx%8)))
            else:
                if os.path.exists(file_addr(decoder,attack,i,j)):        
                    f_lst.append(file_addr(decoder,attack,i,j))    
        cnt+=1
        os.makedirs(os.path.join(task_dir,"%d"%cnt),exist_ok=True)
        path = os.path.join(task_dir,"%d.jpg"%cnt)
        save_lst(f_lst,path)

    task_dir = "demon\\attack"
    cnt=0
    for idx, (i,j) in enumerate(Good_Example):
        f_lst=[]
        f_lst.append(file_addr("imagenet","nat",i,j))
        for attack in ["selfaugall","polygon"]:
            if attack == "selfaugall":
                if os.path.exists(os.path.join("add",file_addr(decoder,attack,idx //8 ,idx%8))):        
                    f_lst.append(os.path.join("add",file_addr(decoder,attack,idx //8 ,idx%8)))
            else:
                if os.path.exists(file_addr(decoder,attack,i,j)):        
                    f_lst.append(file_addr(decoder,attack,i,j))                    
        cnt+=1
        os.makedirs(os.path.join(task_dir,"%d"%cnt),exist_ok=True)
        path = os.path.join(task_dir,"%d.jpg"%cnt)
        save_lst(f_lst,path)        
    
    task_dir = "demon\\meansigma"
    cnt=0
    for i,j in Good_Example:
        f_lst=[]
        f_lst.append(file_addr("imagenet","nat",i,j))
        for attack in ["selfaug_mean","selfaug_sigma"]:
            if os.path.exists(file_addr(decoder,attack,i,j)):        
                f_lst.append(file_addr(decoder,attack,i,j))
        cnt+=1
        os.makedirs(os.path.join(task_dir,"%d"%cnt),exist_ok=True)
        path = os.path.join(task_dir,"%d.jpg"%cnt)
        save_lst(f_lst,path)          
else:
    task_dir = "demon\\model_depth"
    cnt=0
    for i,j in Good_Example:
        attack = "selfaugall"
        f_lst=[]
        f_lst.append(file_addr("imagenet","nat",i,j))
        for decoder in decoder_level:
            if os.path.exists(file_addr(decoder,attack,i,j)):
                f_lst.append(file_addr(decoder,attack,i,j))
        cnt+=1
        os.makedirs(os.path.join(task_dir,"%d"%cnt),exist_ok=True)
        path = os.path.join(task_dir,"%d.jpg"%cnt)
        save_lst(f_lst,path)

    task_dir = "demon\\attack"
    cnt=0
    for i,j in Good_Example:
        f_lst=[]
        f_lst.append(file_addr("imagenet","nat",i,j))
        for attack in ["selfaugall","polygon"]:
            if os.path.exists(file_addr(decoder,attack,i,j)):        
                f_lst.append(file_addr(decoder,attack,i,j))
        cnt+=1
        os.makedirs(os.path.join(task_dir,"%d"%cnt),exist_ok=True)
        path = os.path.join(task_dir,"%d.jpg"%cnt)
        save_lst(f_lst,path)        
    
    task_dir = "demon\\meansigma"
    cnt=0
    for i,j in Good_Example:
        f_lst=[]
        f_lst.append(file_addr("imagenet","nat",i,j))
        for attack in ["selfaug_mean","selfaug_sigma"]:
            if os.path.exists(file_addr(decoder,attack,i,j)):        
                f_lst.append(file_addr(decoder,attack,i,j))
        cnt+=1
        os.makedirs(os.path.join(task_dir,"%d"%cnt),exist_ok=True)
        path = os.path.join(task_dir,"%d.jpg"%cnt)
        save_lst(f_lst,path)          

        
        
