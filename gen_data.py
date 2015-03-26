#!/usr/bin/env python
import os, sys

# Set data paths and names
cmd = "make -C tools"
os.system(cmd)
data_path = 'data/'
cmd = "cd %s;for i in ./*.zip;do unzip -n $i;done"%data_path
print cmd
os.system(cmd)
cmd = "cd %s;for i in ./*.rar;do unrar x -o- $i;done"%data_path
print cmd
os.system(cmd)
cmd = "cd %s;tar zxvf ./Webscope_C14.tgz"%data_path
print cmd
os.system(cmd)
cmd = 'mv %sLearning\ to\ Rank\ Challenge %sYAHOO'%(data_path,data_path)
print cmd
os.system(cmd)
cmd = "cd %sYAHOO/; tar jxvf ltrc_yahoo.tar.bz2"%data_path
print cmd
os.system(cmd)
all_data = ['MSLR','YAHOO_SET1','YAHOO_SET2'] #data needed to be scaled
# Set data paths
# Comment it, if you do not run it.
data_dict = {}
data_dict['MSLR'] = 'Fold1/'
data_dict['YAHOO_SET1'] = 'YAHOO/'
data_dict['YAHOO_SET2'] = 'YAHOO/'
cmd = 'make -C tools/'
os.system(cmd);
data_list = []
for data in all_data:
	if data in data_dict.keys():
		data_list.append(data)
for data in data_list:
	if (data == 'MSLR'):
		pathdata = os.path.join(data_path,data_dict[data],'train.txt')
		pathtest = os.path.join(data_path,data_dict[data],'test.txt')
	elif (data == 'YAHOO_SET1'):
		pathdata = os.path.join(data_path,data_dict[data],'set1.train.txt')
		pathtest = os.path.join(data_path,data_dict[data],'set1.test.txt')
	elif (data == 'YAHOO_SET2'):
		pathdata = os.path.join(data_path,data_dict[data],'set2.train.txt')
		pathtest = os.path.join(data_path,data_dict[data],'set2.test.txt')

	if not os.path.exists("%s.scale"%pathdata) or not os.path.exists("%s.scale"%pathtest):
		if not os.path.exists(pathdata):
			print 'Files of ', data, ' not exist'
			continue
		if ((not os.path.exists('model/%s.scale' % data)) and (not os.path.exists("%s.scale"%pathtest))) or (not os.path.exists("%s.scale"%pathdata)):
			cmd = 'tools/svm-scale -l 0 -u 1 -s model/%s.scale %s > %s.scale' %(data, pathdata, pathdata)
			print cmd
			os.system(cmd)
		if	not os.path.exists("%s.scale"%pathtest):
			cmd = 'tools/svm-scale -l 0 -u 1 -r model/%s.scale %s > %s.scale' %(data, pathtest, pathtest)
			print cmd
			os.system(cmd)
