from  parameter import *
from parse_digit import *
import os
from os import system

cmd = "make -C liblinear >/dev/null 2>/dev/null; make -C tools > /dev/null 2>/dev/null; mkdir log model 2>/dev/null"
system(cmd)

methodlist = ['partial-pairs','l2-loss-ranksvm']
solver = {"partial-pairs":0,"l2-loss-ranksvm":6}

#remove those data you are not interested in its result
data = ['MQ2007-list','MQ2008-list']

print "\\begin{tabular}{l"+"|rrr"*len(data) +"}"
if 'MQ2007-list' in data:
	print "& \\multicolumn{3}{|c}{MQ2007-list}",
if 'MQ2008-list' in data:
	print "& \\multicolumn{3}{|c}{MQ2008-list}",
print "\\\\"
print "& & Training & Pairwise "*len(data),"\\\\"
print "Data set "+"& $C$ & time (s) & accuracy "*len(data) +"\\\\"
print "\\hline"
for method in methodlist:
	print output_name[method],
	o = []
	for d in data:
		dp = log_path + d + '.' + method+ '.pairwise-accuracy.log'
		try:
			tmp_data = open(dp,'r').readlines()
		except:
			if (method == "partial-pairs"):
				traindata = path+data_path[d]+'.partial'
				if not os.path.exists(traindata):
					cmd = "tools/closest_pairs %s > %s"%(path+data_path[d],traindata)
					system('echo \'%s\' >> %s'%(cmd, dp))
					system(cmd)
			else:
				traindata = path+data_path[d]

			model = model_path + d + '.' + method + 'pairwise-accuracy.model'
			testdata = path + test_path[d]

			if method == "partial-pairs":
				cmd = "%s -c %s -s %s %s %s >> %s"%(train_exe, best_partial_c[d],solver[method],traindata,model,dp)
			elif method == "l2-loss-ranksvm":
				cmd = "%s -c %s -s %s %s %s >> %s"%(train_exe, best_c['pairwise-accuracy'][method][d],solver[method],traindata,model,dp)
			system('echo \'%s\' >> %s'%(cmd, dp))
			system(cmd)
			cmd = "%s %s %s tmp_file >> %s;rm -f tmp_file"%(predict_exe, testdata, model, dp)
			system('echo \'%s\' >> %s'%(cmd, dp))
			system(cmd)
			tmp_data = open(dp,'r').readlines()
		
		if method == "partial-pairs":
			o.append(best_partial_c_print[d])
		elif method == "l2-loss-ranksvm":
			o.append(best_full_c_print[d])
		max_iter = 0;
		for l in tmp_data:
			if 'ITERATION' in l:
				max_iter = 1;
			if 'time' in l:
				time = l.split(' ')[-1].strip()
				digit = FormatWithCommas("%5.1f",float(time))
				o.append(digit)
			if 'Accuracy' in l:
				acc = l.split(' ')[-1].strip().strip('%')
				digit = "%5.2f"%float(acc)+"\\%"
				o.append(digit)	
		if max_iter == 1:
			o[-1] = o[-1]+"^*"
			o[-2] = o[-2]+"^*"
	for l in o:
		print "& $%s$ "%l,
	print "\\\\"
print "\\end{tabular}"
