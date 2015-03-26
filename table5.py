from  parameter import *
from parse_digit import *
from os import system

cmd = "make -C liblinear >/dev/null 2>/dev/null;mkdir log model 2>/dev/null"
system(cmd)

#remove those method/data/criterion you are not interested in its result
methodlist = ['l2-loss-ranksvm','l1-loss-svr','l2-loss-svr']
data = ['MQ2007','MQ2008','MSLR','YAHOO_SET1','YAHOO_SET2','MQ2007-list','MQ2008-list']
criteria = ['ndcg','pairwise-accuracy']

solver = {'l2-loss-ranksvm':6,'l1-loss-svr':2,'l2-loss-svr':1}
for cr in criteria:
	print "\\begin{tabular}{l"+"|rr"*len(methodlist) +"}"
	if 'l2-loss-ranksvm' in methodlist:
		print "& \\multicolumn{2}{|c}{L2-loss RankSVM}",
	if 'l1-loss-svr' in methodlist:
		print "& \\multicolumn{2}{|c}{L1-loss SVR}",
	if 'l2-loss-svr' in methodlist:
		print "& \\multicolumn{2}{|c}{L2-loss SVR}",
	print "\\\\"
	if cr == 'ndcg':
		print "& Training &"*len(methodlist),"\\\\"
		print "Data set "+"& time (s) & NDCG "*len(methodlist) +"\\\\"
	else:
		print "& Training & Pairwise "*len(methodlist) +"\\\\"
		print "Data set "+"& time (s) & accuracy "*len(methodlist) +"\\\\"
	print "\\hline"
	for d in data:
		o = []
		if cr == 'ndcg' and (d == 'MQ2007-list' or d == 'MQ2008-list'):
			continue
		for method in methodlist:
			dp = log_path + d + '.' + method+ '.' + cr + '.log'
			try:
				tmp_data = open(dp,'r').readlines()
			except:
				traindata = path + data_path[d]
				model = model_path + d + '.' + method + '.' + cr + '.model'
				testdata = path + test_path[d]
				cmd = "%s -c %s -s %s %s %s >> %s"%(train_exe, best_c[cr][method][d],solver[method],traindata,model,dp)
				system('echo \'%s\' >> %s'%(cmd, dp))
				system(cmd)
				cmd = "%s %s %s tmp_file >> %s;rm -f tmp_file"%(predict_exe, testdata, model, dp)
				system('echo \'%s\' >> %s'%(cmd, dp))
				system(cmd)

				tmp_data = open(dp,'r').readlines()
			max_iter = 0;
			for l in tmp_data:
				if 'ITERATION' in l:
					max_iter = 1;
				if 'time' in l:
					time = l.split(' ')[-1].strip()
					digit = FormatWithCommas("%5.1f",float(time))
					o.append(digit)
				if cr == 'pairwise-accuracy':
					if 'Accuracy' in l:
						acc = l.split(' ')[-1].strip().strip('%')
						digit = "%5.2f"%float(acc)+"\\%"
						o.append(digit)	
				elif cr == 'ndcg':
					if d == 'YAHOO_SET1' or d == 'YAHOO_SET2':
						if '(YAHOO)' in l:
							ndcg = l.split(' ')[-1].strip()
							digit = "%1.4f"%float(ndcg)
							o.append(digit)	
					else:
						if '(LETOR)' in l:
							ndcg = l.split(' ')[-1].strip()
							digit = "%1.4f"%float(ndcg)
							o.append(digit)	
			if max_iter == 1:
				o[-1] = o[-1]+"^*"
				o[-2] = o[-2]+"^*"
		print output_name[d],
		for l in o:
			print "& $%s$ "%l,
		print "\\\\"
	print "\\end{tabular}"
