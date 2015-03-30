from  parameter import *
import os
from os import system,popen

#remove those method/data you are not interested in the following two lines
methodlist = ['y-avltree']#methodlist = ['direct-count','y-rbtree','wx-rbtree','selectiontree','y-avltree','y-aatree']
data = ['MQ2007','MQ2008']#data = ['MSLR','YAHOO_SET1','MQ2007-list','MQ2008-list']

cmd = "make -C liblinear >/dev/null 2>/dev/null; mkdir log 2>/dev/null; mkdir fig4 2>/dev/null"
system(cmd)

solver = {'direct-count':3,'y-rbtree':4,'wx-rbtree':5,'selectiontree':6,'y-avltree':7,'y-aatree':8}
f = popen('wc -l fig4.m')#wc -l fig4.m : return the total lines of the file fig4.m; f :virtual file handle
s = int(f.readline().split()[0])# s: total lines of file fig4.m

f.close()
tmp = []
for d in data:
	if not os.path.exists('fig4/%s.fig4.png'%d):
		tmp.append(d)
data = tmp
if len(data) > 0 and len(methodlist) > 0:
	inputlist = "inputlist = {'%s'"%data[0]	
	for d in xrange(1,len(data)):
		inputlist = inputlist+",'%s'"%data[d]
	inputlist = inputlist + "};\n"	
	solverlist = "solver = {'%s'"%methodlist[0]
	for t in xrange(1,len(methodlist)):
		solverlist = solverlist + ",'%s'"%methodlist[t]
	solverlist = solverlist + "};\n"
	p = open("./tmp.m","w")
	p.write(inputlist)
	p.write(solverlist)
	p.close()
	#copy the file fig4.m to tmp.m except the first two line,the first two lines are inputlist(ie. data) and solverlist(ie. methodlist) we configurate in this file
	cmd = "tail -n %s ./fig4.m >> ./tmp.m"%(s-2)
	system(cmd)
	system("rm -f fig4.m; mv tmp.m fig4.m")
for d in data:
	for method in methodlist:
		dp = log_path + d + '.' + method+ '.fig4.log'
		try:
			tmp_data = open(dp,'r').readlines()
		except:
			traindata = path + data_path[d]
			cmd = "%s -c 1 -e 1e-20 -s %s %s /dev/null >> %s"%(train_exe,solver[method],traindata,dp)
			system('echo \'%s\' >> %s'%(cmd, dp))
			system(cmd)
if len(data) > 0:
	cmd = "matlab -nodisplay -nodesktop -r \"fig4;exit\""
	system(cmd)

