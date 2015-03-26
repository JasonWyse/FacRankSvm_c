from  parameter import *
import os
from os import system,popen

#remove those method/data/criterion you are not interested in its result
methodlist = ['tree-tron','prsvm+','treeranksvm']
data = ['MQ2007','MQ2008','MSLR','YAHOO_SET1','YAHOO_SET2','MQ2007-list','MQ2008-list']
criteria = ['pairwise-accuracy','ndcg']

m_file = {'pairwise-accuracy':'acc','ndcg':'ndcg'}
tmp = {}
tmp['pairwise-accuracy'] = []
tmp['ndcg'] = []

for d in data:
	if (not os.path.exists('fig5_6/%s.fun-value.png'%d) or (not os.path.exists('fig5_6/%s.accuracy.png'%d))) and 'pairwise-accuracy' in criteria:
		tmp['pairwise-accuracy'].append(d)
	if not os.path.exists('fig5_6/%s.ndcg.png'%d) and d != 'MQ2007-list' and d != 'MQ2008-list' and 'ndcg' in criteria:
		tmp['ndcg'].append(d)
data = tmp

cmd = "make -C liblinear >/dev/null 2>/dev/null; make -C tools >/dev/null 2>/dev/null; ln -sf tools/eval .; mkdir log 2>/dev/null; mkdir fig5_6 2>/dev/null"
system(cmd)

for cr in criteria:
	if len(data[cr]) > 0 and len(methodlist) > 0:
		inputlist = "inputlist = {"
		inputlist = inputlist+"'%s'"%data[cr][0]
		f = popen('wc -l %s.m'%m_file[cr])
		s = int(f.readline().split()[0])
		f.close()
		for d in xrange(1,len(data[cr])):
			inputlist = inputlist+",'%s'"%data[cr][d]
		inputlist = inputlist + "};\n"
		solverlist = "solver = {"
		solverlist = solverlist+"'%s'"%methodlist[0]
		for t in xrange(1,len(methodlist)):
			solverlist = solverlist + ",'%s'"%methodlist[t]
		solverlist = solverlist + "};\n"
		p = open("./tmp.m","w")
		p.write(inputlist)
		p.write(solverlist)
		p.close()
		cmd = "tail -n %s ./%s.m >> ./tmp.m"%(s-2,m_file[cr])
		system(cmd)
		system("rm -f %s; mv tmp.m %s.m"%(m_file[cr],m_file[cr]))
	for d in data[cr]:
		for method in methodlist:
			dp = log_path + d + '.' + method + '.' + cr + '.log'
			try:
				tmp_data = open(dp,'r').readlines()
			except:
				traindata = path + data_path[d]
				testdata = path + test_path[d]
				if method == 'tree-tron':
					cmd = "%s -c %s -e 1e-20 -s 6 %s %s >> %s"%(train_fig56_exe,best_c[cr]['l2-loss-ranksvm'][d],traindata,testdata,dp)
				elif method == 'prsvm+':
					cmd = "%s -c %s -e 1e-20 -s 9 %s %s >> %s"%(train_fig56_exe,best_c[cr]['l2-loss-ranksvm'][d],traindata,testdata,dp)
				elif method == 'treeranksvm':
					if d not in best_c[cr]['l1-loss-ranksvm'].keys():
						continue
					conf_file = conf_path + d + '.treeranksvm.' + cr + '.cfg'
					if not os.path.exists(conf_file):
						f = open(conf_file,'w')
						f.writelines('[Parameters]\nepsilon=1e-6\nmax_iterations=100000\n')
						f.write('regparam=%s\n'%(1.0/(best_c[cr]['l1-loss-ranksvm'][d]*num_pairs[d])))
						f.writelines('[Input]\ntrain_set=%s\n'%traindata)
						f.write('test_set=%s\n'%testdata)
						f.close()
					cmd = "python %s %s >> %s"%(treeranksvm_exe,conf_file,dp)
				system('echo \'%s\' >> %s'%(cmd, dp))
				system(cmd)
	if len(data[cr]) > 0:
		cmd = "matlab -nodisplay -nodesktop -r \"%s;exit\""%m_file[cr]
		system(cmd)
