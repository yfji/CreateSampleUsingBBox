import os
import cv2
import numpy as np
import cPickle
from PIL import Image
import xml.etree.ElementTree as ET
from visualize import vis_square

_max_overlaps=0.09
_max_allowed_overlap=0.3
_cropped_ratio=0.5
benchmark_root='I:/DeepLearn/benchmark'

def calc_roi_overlaps(rois, gt_boxes, top_k):
    max_overlaps=np.zeros(rois.shape[0])
    max_overlaps.fill(1000)
    for k in xrange(rois.shape[0]):
        roi_w,roi_h=rois[k][2]-rois[k][0],rois[k][3]-rois[k][1]
        area=float(roi_w*roi_h)
        overlaps=np.zeros(gt_boxes.shape[0])
        x=np.maximum(rois[k][0],gt_boxes[:,0])
        y=np.maximum(rois[k][1],gt_boxes[:,1])
        xx=np.minimum(rois[k][2], gt_boxes[:,2])
        yy=np.minimum(rois[k][3],gt_boxes[:,3])
        w,h=np.maximum(0,xx-x),np.maximum(0,yy-y)
        inters=(w*h).astype(np.float32)
        overlaps=inters/area
        max_overlaps[k]=np.max(overlaps)
    valid_inds=np.where(max_overlaps<_max_overlaps)[0]
    keep=np.random.choice(valid_inds,min(top_k, valid_inds.size), replace=False)
    return keep

def create_roidb(label_path, cache_path, min_scale):
    roidb=[]
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as fid:
             roidb=cPickle.load(fid)
        return roidb
    with open(label_path, 'r') as label_file:
        lines=label_file.readlines() 
    roidb_item={}
    """(x1,y1,x2,y2,area,label)"""
    boxes=np.zeros((0,6))
    num_images=0
    for i in xrange(len(lines)):
        line=lines[i].rstrip()
        if line[-4:]=='.bmp':
            num_images+=1
            if num_images>=2:
                roidb_item['gt_boxes']=boxes
                roidb.append(roidb_item)
                roidb_item={}
                roidb_item['image_path']=line
                boxes=np.zeros((0,6))
            if num_images==1:
                roidb_item['image_path']=line
        elif len(line)>0:
            items=line.split(' ')
            roi=[float(item) for item in items]
            roi.append(roi[2]*roi[3])
            roi.append(1.0)
            roi=np.asarray(roi)
            roi[2]=roi[2]+roi[0]
            roi[3]=roi[3]+roi[1]
            boxes=np.vstack((boxes, roi))           
    roidb_item['gt_boxes']=boxes
    roidb.append(roidb_item)
    create_hard_examples(roidb, min_scale)
    with open(cache_path, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    return roidb
    
def create_hard_examples(roidb, min_scale, neg_per_image=4):
    """ratio=height/width"""
    enum_size=3
    #aspect_ratio=[1.0, 1.0, 1.0]
    scale=[0.9,1.0,1.1]
    shiftx=[-1,0,1,0]
    shifty=[0,-1,0,1]
    neg_per_roi=5*neg_per_image
    for entry in roidb:
        path=entry['image_path']
        image=Image.open(path)
        im_width=image.size[0]; im_height=image.size[1]
        gt_boxes=entry['gt_boxes']
        #print(rois)
        too_large=0
        for i in xrange(gt_boxes.shape[0]):
            if gt_boxes[i][4]>0.4*(im_width*im_height):
                too_large=1; break
        if too_large:continue
        neg_rois=np.zeros((0,6))
        for i in xrange(gt_boxes.shape[0]):
            r=gt_boxes[i]
            if r[4]<min_scale*im_width*im_height:
                continue
            pad=np.asarray([r[0], r[1], im_width-r[2], im_height-r[3]])
            roi=np.asarray([r[2]-r[0], r[3]-r[1], r[2]-r[0], r[3]-r[1]])
            shiftable=np.where((pad>np.round((1-_max_allowed_overlap)*roi+0.6))==1)[0]
            for i in xrange(shiftable.shape[0]):
                s=shiftable[i]
                for k in xrange(neg_per_roi):
                    if int((1-_max_allowed_overlap)*roi[s])>=int(pad[s]):
                        print(path)
                        print((1-_max_allowed_overlap)*roi[s],pad[s])
                    deltax=shiftx[s]*np.random.randint(int((1-_max_allowed_overlap)*roi[s]),int(pad[s]))
                    deltay=shifty[s]*np.random.randint(int((1-_max_allowed_overlap)*roi[s]),int(pad[s]))
                    if deltax==0:
                        if pad[0]>0 and pad[2]>0:
                            deltax=np.random.randint(-pad[0], pad[2])
                    if deltay==0:
                        if pad[1]>0 and pad[3]>0:
                            deltay=np.random.randint(-pad[1],pad[3])
                    neg_roi=np.asarray([r[0]+deltax, r[1]+deltay, r[2]-r[0], r[3]-r[1]])
                    #neg_roi=np.asarray([r[0]+deltax, r[1]+deltay, r[2]+deltax, r[3]+deltay])

                    neg_roi[2:4]*=scale[np.random.randint(0,enum_size-1)]
                    #neg_roi[3]=neg_roi[2]*aspect_ratio[np.random.randint(0,enum_size-1)]
                    last_two=np.asarray([neg_roi[2]*neg_roi[3], 0.0])
                    neg_roi[2:4]+=neg_roi[0:2]
                    neg_roi=np.hstack((neg_roi,last_two))
                    neg_rois=np.vstack((neg_rois,neg_roi))
        neg_rois=np.maximum(0,neg_rois)
        neg_rois[:,2]=np.minimum(im_width, neg_rois[:,2])
        neg_rois[:,3]=np.minimum(im_height, neg_rois[:,3])
        small_overlaps_inds=calc_roi_overlaps(neg_rois, gt_boxes, neg_per_image)
        entry['gt_boxes']=np.vstack((entry['gt_boxes'],neg_rois[small_overlaps_inds]))

def save_images(roidb, pos_root, neg_root):
    pos_cnt=0
    neg_cnt=0
    for db in roidb:
        path=db['image_path']
        image=cv2.imread(path)
        rois=db['gt_boxes']
        for i in xrange(rois.shape[0]):
            roi=rois[i]
            if roi[-1]>0.5:
                roi_image=image[int(roi[1]):int(roi[3]),int(roi[0]):int(roi[2]),:]
                cv2.imwrite(os.path.join(pos_root,'person_%d.jpg'%pos_cnt),roi_image)
                pos_cnt+=1
            else:
                roi_image=image[int(roi[1]):int(roi[3]),int(roi[0]):int(roi[2]),:]
                cv2.imwrite(os.path.join(neg_root,'nonperson_%d.jpg'%neg_cnt),roi_image)
                neg_cnt+=1
    print('done')

def save_neg_examples(image_root, other_db='voc2007'):
    db_root_dict={'voc2007':os.path.join(benchmark_root,'VOCdevkit2007/VOC2007')}
    db_root=db_root_dict[other_db]
    txt=os.path.join(db_root,'ImageSets/Main/val.txt')
    xmls=os.path.join(db_root,'Annotations')
    jpgs=os.path.join(db_root,'JPEGImages')
    txt_file=open(txt,'r')
    image_titles=txt_file.readlines()
    txt_file.close()
    cnt=3600
    for i in xrange(len(image_titles)):
        title=image_titles[i].rstrip()
        xml=os.path.join(xmls,title+'.xml')
        tree = ET.parse(xml)
        objs=tree.findall('object')
        has_pot=0
        for ix,obj in enumerate(objs):
            name=str(obj.find('name').text)
            if 'pot' in name:
                print(name+': containing pot')
                has_pot=1;break
        if has_pot:
            continue
        image=cv2.imread(os.path.join(jpgs,title+'.jpg'))
        im_width,im_height=image.shape[1],image.shape[0]
        cropped_dim=_cropped_ratio*min(im_width,im_height)
        center=np.asarray([(1.0*im_width-cropped_dim)/2, (1.0*im_height-cropped_dim)/2, (1.0*im_width+cropped_dim)/2, (1.0*im_height+cropped_dim)/2]).astype(np.int32)
        roi=image[center[1]:center[3],center[0]:center[2],:]
        cv2.imwrite(os.path.join(image_root, 'nonpot_%d.jpg'%cnt),roi)
        cnt+=1
    print('done')

def create_list(label_file, test_file, pos_root, neg_root):
    pos_samples=os.listdir(pos_root)
    neg_samples=os.listdir(neg_root)
    pos_test_num=0.2*len(pos_samples)
    neg_test_num=0.2*len(neg_samples)
    pos_index=np.random.permutation(np.arange(len(pos_samples)))
    neg_index=np.random.permutation(np.arange(len(neg_samples)))
    tf=open(test_file, 'w')
    lf=open(label_file, 'w')
    for ix in pos_index:
        name=pos_samples[ix]
        name=os.path.join(pos_root, name)
        if ix<pos_test_num:
            tf.write(name+' 1\n')
        else:
            lf.write(name+' 1\n')
    for ix in neg_index:
        name=neg_samples[ix]
        name=os.path.join(neg_root, name)
        if ix<neg_test_num:
            tf.write(name+' -1\n')
        else:
            lf.write(name+' -1\n')
    tf.close()
    lf.close()

def create_hard_examples(hard_test_file, neg_root):
    neg_samples=os.listdir(neg_root)
    tf=open(hard_test_file, 'w')
    for name in neg_samples:
        name=os.path.join(neg_root, name)
        tf.write(name+' -1\n')
    tf.close()

def show_some_db(roidb, max_size):
    num=25
    start=100
    blob=np.zeros((num,3,max_size,max_size))
    for i in xrange(start,start+num):
        entry=roidb[i]
        img_path=entry['image_path']
        image=cv2.imread(img_path)
        boxes=entry['gt_boxes'].astype(np.int32)
        for r in xrange(boxes.shape[0]):
            box=boxes[r]
            if box[-1]>0.5:
                color=(0,255,255)
            else:
                color=(255,100,0)
            cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), color, 2)
        roi=cv2.resize(image, (max_size,max_size), interpolation=cv2.INTER_LINEAR)
        blob[i-start,:,:,:]=roi.transpose(2,0,1)[...]
    vis_square(blob, save_path='./db_part.jpg')
    
if __name__=='__main__':
    label_path='people_bbox.txt'
    svm_label_file='person_label.txt'
    svm_test_file='test_label.txt'
    svm_hard_file='hard_label.txt'
    cache_path='cache/people.pkl'
    pos_root='I:/TestOpenCV/Images/PedestrainBenchmark/samples/people'
    neg_root='I:/TestOpenCV/Images/PedestrainBenchmark/samples/nonpeople'
    hard_root='I:/TestOpenCV/Images/PedestrainBenchmark/samples/hardexample/crop'
    roidb=create_roidb(label_path, cache_path, 1e-3)
    show_some_db(roidb, 150)
    create_hard_examples(svm_hard_file, hard_root)
    #save_images(roidb, pos_root, neg_root)
    #save_neg_examples(neg_root)
    #create_list(svm_label_file, svm_test_file, pos_root, neg_root);
