from  parameter import *
from parse_digit import *
from os import system

cmd = "make -C tools >/dev/null 2>/dev/null;mkdir log model 2>/dev/null"
system(cmd)

#remove those method/data you are not interested in its result
methodlist = ['random-forest','gbdt']
data = ['MQ2007','MQ2008','MSLR','YAHOO_SET1','YAHOO_SET2','MQ2007-list','MQ2008-list']

print "\\begin{tabular}{l"+"|rrr"*len(methodlist) +"}"
if 'random-forest' in methodlist:
	print "& \\multicolumn{3}{|c}{Random forests}",
if 'gbdt' in methodlist:
	print "& \\multicolumn{3}{|c}{GBDT}",
print "\\\\"
print "& Training & Pairwise & "*len(methodlist), "\\\\"
print "Data set "+"& time (s) & accuracy & NDCG "*len(methodlist) +"\\\\"
print "\\hline"
for d in data:
	o = []
	for method in methodlist:
		dp = log_path + d + '.' + method+ '.fewtrees.log'
		try:
			tmp_data = open(dp,'r').readlines()
		except:
			traindata = path + data_path[d]
			testdata = path + test_path[d]
			if method == 'random-forest':
				cmd = "%s -f %s -F -z -p %s -k %s -t %s %s %s ./tmp_file >> %s 2>/dev/null"%(tree_exe,num_feature[d],num_processors, num_sampled_feature[d], tree_num_few[method],traindata,testdata,dp)
			elif method == 'gbdt':
				model = model_path + d + '.' + method + '.' + 'fewtrees.model'
				cmd = "mpirun -np %s %s %s %s %s 4 100 0.1 -m >%s 2>> %s"%(8,gbrt_exe,traindata,num_instance[d],num_feature[d]+1,model,dp)
				system('echo \'%s\' >> %s'%(cmd, dp))
				system(cmd)
				cmd = "cat %s|python %s ./tmp_exe"%(model,gbrt_compile_test)
				system('echo \'%s\' >> %s'%(cmd, dp))
				system(cmd)
				cmd = "cat %s|./tmp_exe > ./tmp_file"%testdata
			system('echo \'%s\' >> %s'%(cmd, dp))
			system(cmd)
			cmd = "tools/eval ./tmp_file %s >> %s;rm -f tmp_file ./tmp_exe*"%(testdata, dp)
			system('echo \'%s\' >> %s'%(cmd, dp))
			system(cmd)
			tmp_data = open(dp,'r').readlines()
		for l in tmp_data:
			if 'time' in l:
				time = l.split(' ')[-1].strip()
				digit = FormatWithCommas("%5.1f",float(time))
				digit = "$"+digit+"$"
				o.append(digit)
			if 'accuracy' in l:
				acc = l.split(' ')[-2].strip().strip('%')
				digit = "$%5.2f$"%float(acc)+"\\%"
				o.append(digit)	
			if d == 'YAHOO_SET1' or d == 'YAHOO_SET2':
				if '(YAHOO)' in l:
					ndcg = l.split(' ')[-2].strip()
					digit = "$%1.4f$"%float(ndcg)
					o.append(digit)	
			else:
				if 'Mean' in l:
					if  d == 'MQ2007-list' or d == 'MQ2008-list':
						digit = "NA"
					else:
						ndcg = l.split(' ')[-2].strip()
						digit = "$%1.4f$"%float(ndcg)
					o.append(digit)	
	print output_name[d],
	for l in o:
		print "& %s "%l,
	print "\\\\"
print "\\end{tabular}"
