from  parameter import *
import os
from os import system,popen

#remove those method/data you are not interested in the following two lines
methodlist = ['tree-tron','prsvm+','treeranksvm']
data = ['MQ2007','MQ2008','YAHOO_SET2','MQ2007-list','MQ2008-list']
C_list=[100,1,1e-4]

cmd = "make -C liblinear >/dev/null 2>/dev/null; make -C tools >/dev/null 2>/dev/null; ln -sf tools/eval .; mkdir log 2>/dev/null; mkdir fig7 2>/dev/null"
system(cmd)

f = popen('wc -l fig7.m')
s = int(f.readline().split()[0])
f.close()
tmp = []
for d in data:
	for c in C_list:
		if not os.path.exists('fig7/%s-c%s.png'%(d,c)) and not d in tmp:
			tmp.append(d)
data = tmp
if len(data) > 0 and len(methodlist) > 0:
	inputlist = "inputlist = {"
	for d in xrange(0,len(data)):
		for c in C_list:
			inputlist = inputlist+"'%s-c%s',"%(data[d],c)
	inputlist = inputlist[:-1] + "};\n"
	solverlist = "solver = {'%s'"%methodlist[0]
	for t in xrange(1,len(methodlist)):
		solverlist = solverlist + ",'%s'"%methodlist[t]
	solverlist = solverlist + "};\n"
	p = open("./tmp.m","w")
	p.write(inputlist)
	p.write(solverlist)
	p.close()
	cmd = "tail -n %s ./fig7.m >> ./tmp.m"%(s-2)
	system(cmd)
	system("rm -f fig7.m; mv tmp.m fig7.m")
for d in data:
	for method in methodlist:
		for c in C_list:
			dp = log_path + d + '-c' + str(c) + '.'+ method+ '.log'
			try:
				tmp_data = open(dp,'r').readlines()
			except:
				traindata = path + data_path[d]
				testdata = path + test_path[d]
				if method == 'tree-tron':
					cmd = "%s -c %s -e 1e-20 -s 6 %s /dev/null >> %s"%(train_exe,c,traindata,dp)
				elif method == 'prsvm+':
					cmd = "%s -c %s -e 1e-20 -s 9 %s /dev/null >> %s"%(train_exe,c,traindata,dp)
				elif method == 'treeranksvm':
					conf_file = conf_path + d + '.treeranksvm.' + 'c' + str(c) + '.cfg'
					if not os.path.exists(conf_file):
						f = open(conf_file,'w')
						f.writelines('[Parameters]\nepsilon=1e-6\nmax_iterations=100000\n')
						f.write('regparam=%s\n'%(1.0/(c*num_pairs[d])))
						f.writelines('[Input]\ntrain_set=%s\n'%traindata)
						f.write('test_set=%s\n'%testdata)
						f.close()
					cmd = "python %s %s >> %s"%(treeranksvm_exe,conf_file,dp)
				system('echo \'%s\' >> %s'%(cmd, dp))
				system(cmd)
	if len(data) > 0:
		cmd = "matlab -nodisplay -nodesktop -r \"fig7;exit\""
		system(cmd)
