from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from sklearn.metrics import roc_auc_score
import sklearn


repeat_time=1
base_path="../data"
#directory_number=['1458','2259','2261','2821','2997','3358','3386','3427','3476','yoyi-5300','uncensored_support','lastfm']
save_path="output"
directory_number=['2259']
train_yzbx=['uncensored_support','lastfm']
train_file="train.bid.txt"   #the format for iPinYou and yoyi-5300
train_file2="train.yzbx.txt" #the format for support
test_file="test.yzbx.txt"

def my_log(a):
	if  a<=0:
		#countt.counttt+=1
		return math.log(1e-10)
	else:
		return math.log(a)

def win_prob(bid,zw_dict):
    if bid in zw_dict:
        return zw_dict[bid]
    last_key = -1
    for key in zw_dict:
        if last_key == -1:
            last_key = key
        if bid <= key:
            return zw_dict[last_key]
        else:
            last_key = key
    return 1.

def train(num,base_path=base_path):
	"""
	this function is used to process train.bid.txt format
	num:sequence number of the iPinYou,str
	"""
	train_data_file=os.path.join(base_path,num,train_file)
	b_data=defaultdict(list)
	fi=open(train_data_file,'r')
	size=0
	maxb=0
	for line in fi:
		s=line.strip().split()
		b=int(s[0])
		maxb= max(b,maxb)
		o=int(s[1])
		b_data[b].append(o)
		size+=1
	fi.close()
	b_data=sorted(b_data.items(),key=lambda e:e[0],reverse=False)
	b_data=dict(b_data)

	bdns=[]
	wins=0
	for z in b_data:
		wins=sum(b_data[z])
		b=z
		d=wins
		n=size
		bdn=[b,d,n]
		bdns.append(bdn)
		size-=len(b_data[z])

	#print(bdns)

	zw_dict={}
	min_p_w=0
	bdns_length=len(bdns)
	count=0
	p_l_tmp=1.0

	for bdn in bdns:
		count+=1
		b=float(bdn[0])
		d=float(bdn[1])
		n=float(bdn[2])
		if count<bdns_length:
			p_l_tmp*=(n-d)/n
		p_l=p_l_tmp
		p_w=max(1.0-p_l,min_p_w)
		zw_dict[int(b)]=p_w

	#print(zw_dict)
	return zw_dict,maxb




def train2(num,base_path=base_path):
	"""
	this function is used to process train.yzbx.txt format
	"""
	#train_data_file="/home/zyyang/RS/train.yzbx.txt"
	train_data_file=os.path.join(base_path,num,'train.yzbx.txt')
	b_data=defaultdict(list)
	fi=open(train_data_file,'r')
	size=0
	maxb=0
	for line in fi:
		s=line.strip().split()
		b=int(s[2])
		maxb= max(b,maxb)
		o=b>int(s[1])
		o=int(o)
		b_data[b].append(o)
		size+=1
	fi.close()
	b_data=sorted(b_data.items(),key=lambda e:e[0],reverse=False)
	b_data=dict(b_data)

	bdns=[]
	wins=0
	for z in b_data:
		wins=sum(b_data[z])
		b=z
		d=wins
		n=size
		bdn=[b,d,n]
		bdns.append(bdn)
		size-=len(b_data[z])

	zw_dict={}
	min_p_w=0
	bdns_length=len(bdns)
	count=0
	p_l_tmp=1.0

	for bdn in bdns:
		count+=1
		b=float(bdn[0])
		d=float(bdn[1])
		n=float(bdn[2])
		if count<bdns_length:
			p_l_tmp*=(n-d)/n
		p_l=p_l_tmp
		p_w=max(1.0-p_l,min_p_w)
		zw_dict[int(b)]=p_w

	#print(zw_dict)
	return zw_dict,maxb

def draw(num,zw_dict,maxb):
	"""
		draw the survival rate curve
	"""
	b_full_data=[]

	for i in range(1,maxb+1):
		b_full_data.append(win_prob(i,zw_dict))

	s=range(1,maxb+1)
	A,=plt.step(s, b_full_data, 'r-',where='post',label=num,linewidth=1.0)
	font1 = {'family' : 'Times New Roman',
	'weight' : 'normal',
	'size'   : 10,
	}

	tmp_data=np.array(b_full_data)
	tmp_data=1-tmp_data

	legend = plt.legend(handles=[A],prop=font1)
	plt.ylim((0,1))
	
	plt.savefig(save_path+"/km_"+num)
	plt.close(1)
	

def test(num,zw_dict,maxb,base_path=base_path):
	"""

	"""
	count=0
	p_z=0
	log_loss=0
	c_index=0
	#test_data_file=os.path.join(base_path,num,test_file)
	#test_data_file="/home/zyyang/RS/test.yzbx.txt"
	#test_data_file="/home/zyyang/RS/code/support1/test.yzbx.txt"
	test_data_file=os.path.join(base_path,num,test_file)
	fi=open(test_data_file,'r')
	res_data=[]
	label=[]
	pred=[]
	for line in fi:
		count+=1
		s=line.strip().split()
		test_b=int(s[2])
		true_z=int(s[1])
		market_price=int(s[1]) # is the true market price
		win_proba=win_prob(test_b,zw_dict)
		true_prob=win_prob(true_z,zw_dict)
		pred.append(win_proba)
		next_bid=true_z+1
		if next_bid>=maxb:
			#win_prob(next_bid,zw_dict)>=win_prob(maxb,zw_dict):
			next_bid=next_bid
		else:
			while(next_bid not in zw_dict):
				next_bid+=1;
		next_prob=win_prob(next_bid,zw_dict)
		p_z-=my_log(abs(next_prob-true_prob))
		w= test_b>market_price
		label.append(w)
		log_loss-=(w*my_log(win_proba)+(1-w)*my_log(1-win_proba))
	#print(len(label))
	#print(len(pred))
	mytest=[]
	for i in range(0,len(label)):
		mytest.append((pred[i],label[i]))
	mytest=sorted(mytest,key=lambda e:e[0])
	#print(len(mytest))

	c_index=roc_auc_score(label,pred)
	p_z=p_z/count
	log_loss=log_loss/count
	print(("test count is {0}").format(count))
	fi.close()
	return p_z,log_loss,c_index

if __name__=='__main__':
	#test_result=[]
	test_p_z=[]
	test_log_loss=[]
	test_c=[]
	#tmp_result=[]
	final_path=os.path.join(save_path,'km_result.txt')
	fi=open(final_path,'w')

	fi.write('dataset'+'\t'+'anlp'+'\t'+'log_loss'+'\t'+'c_index'+'\n')

	for num in directory_number:
		if num in train_yzbx:
			zw_dict,maxb=train2(num)
		else:
			zw_dict,maxb=train(num)
		for i in range(0,repeat_time):	
			p_z,log_loss,c_index=test(num,zw_dict,maxb)
			test_p_z.append(p_z)
			test_log_loss.append(log_loss)
			test_c.append(c_index)
		test_p_z=np.array(test_p_z)
		test_log_loss=np.array(test_log_loss)
		test_c=np.array(test_c)
		p_z=np.mean(test_p_z)
		log_loss=np.mean(test_log_loss)
		c_index=np.mean(test_c)
		test_p_z=[]
		test_log_loss=[]
		test_c=[]
		#draw(num,zw_dict,maxb)
		fi.write(num+'\t'+str(p_z)+'\t'+str(log_loss)+'\t'+str(c_index)+'\n')
		print('---------------------------------------------------------------------------')
		print("--------------------------------"+num+"-----------------------------------")
		print("anlp: {}".format(p_z))
		print("log loss: {}".format(log_loss))
		print("c-index: {}".format(c_index))
		print('---------------------------------------------------------------------------')
	fi.close()
