inputlist = {'MQ2007','MQ2008','MSLR','YAHOO_SET1','YAHOO_SET2','MQ2007-list','MQ2008-list'};
solver = {'tree-tron','prsvm+','treeranksvm'};

eta0=1e-4;
for i = 1:length(inputlist)
	stop=zeros(1,length(solver));
	touched=zeros(1,length(solver));
	temp=0;
	time = 0;
	l1=0;
	l2=0;
	success=zeros(1,length(solver));
	for s = 1:length(solver)
		counter(s)=1;
		sec(s,1)=0;
		fid = fopen(['log/',inputlist{i},'.',solver{s},'.pairwise-accuracy.log'],'r');
		if (fid == -1)
			break;
		end
		success(s)=1;
		if strcmp(solver{s},'tree-tron')
			l2 = l2+1;
		elseif strcmp(solver{s},'prsvm+')
			l2 = l2+1;
		else
			l1 = l1 + 1;
		end
		line = fgetl(fid);
		while(ischar(line))
			t = strread(line,'%s','delimiter',' \t');
			if (length(t) < 1)
				break;
			end
			if (strcmp(t{1},'iter'))
				counter(s)=counter(s)+1;
				for j = 2:length(t)
					if (strcmp(t{j},'act'))
						act = str2double(t{j+1});
					elseif (strcmp(t{j},'pre'))
						pre = str2double(t{j+1});
					elseif (strcmp(t{j},'f'))
						fun(s,counter(s)-1)=str2double(t{j+1});
					elseif (touched(s) == 0 && strcmp(t{j},'|g|'))
						if (counter(s)== 2)
							temp = str2double(t{j+1});
						elseif (str2double(t{j+1})*1000.0<temp)
							stop(s)=counter(s);
							touched(s)=1;
						end
					end
				end
			elseif (touched(s) == 0 && strcmp(t{1},'epsilon'))
				if (str2double(t{3})<=0.001)
					touched(s)=1;
					stop(s)=counter(s);
				end
			elseif (strcmp(t{1},'Time'))
				sec(s,counter(s))=str2double(t{2});
			elseif (strcmp(t{1},'Pairwise'))
				accuracy(s,counter(s)) = str2double(t{4}(1:length(t{4})-1));
			end
			line =  fgetl(fid);
		end
		if (counter(s)==1)
			break;
		end
		fun(s,counter(s)) = fun(s,counter(s)-1);
		if (strcmp('treeranksvm',solver{s}) || act>=eta0*pre)
			fun(s,counter(s)) = fun(s,counter(s)) - act;
		end
		opttemp(s) = min(fun(s,1:counter(s)));
		time = max(time,sec(s,counter(s)));
		fclose(fid);
	end
	if (l2 > 0)
		opt = min(opttemp(1:l2));
		y = (fun-opt)/opt;
	end
	if (l1 > 0)
		l1opt = opttemp(length(solver));
		y1 = (fun(length(solver),:)-l1opt)/l1opt;
	end
	h = figure;
	set(gca,'fontsize',24);
	for s=1:length(solver)
		if success(s) == 0
			continue;
		end
		if strcmp(solver{s},'tree-tron')
			loglog(sec(s,1:counter(s)),y(s,1:counter(s)),'c-','LineWidth',15);
		elseif strcmp(solver{s},'prsvm+')
			loglog(sec(s,1:counter(s)),y(s,1:counter(s)),'k+:','LineWidth',7);
		else
			loglog(sec(s,1:counter(s)),y1(1:counter(s)),'r--','LineWidth',7);
		end
		if s == 1
			hold on;
		end
	end
	for s=1:length(solver)
		if success(s) == 0
			continue;
		end
		if strcmp(solver{s},'tree-tron')
			loglog(get(gca,'xlim'),[(fun(s,stop(s)-1)-opt)/opt (fun(s,stop(s)-1)-opt)/opt],'c-*','LineWidth',3);
		end
	end
	xlabel('Training Time (Seconds)','FontSize',30);
	ylabel('Difference to Opt. Fun. Value','FontSize',30);
	le=legend(solver);
	set (le,'FontSize',24);
	print(h,'-dpng',['fig5_6/',inputlist{i},'.fun-value.png']);
	clear h;
	set(gca,'fontsize',24);
	for s = 1:length(solver)
		if success(s) == 0
			continue;
		end
		if (sec(s,counter(s))<time)
			counter(s) = counter(s)+1;
			sec(s,counter(s))=time;
			accuracy(s,counter(s)) = accuracy(s,counter(s)-1);
		end
	end
	h = figure;
	set(gca,'fontsize',24);
	for s=1:length(solver)
		if success(s) == 0
			continue;
		end
		if strcmp(solver{s},'tree-tron')
			semilogx(sec(s,2:counter(s)),accuracy(s,2:counter(s)),'c-','LineWidth',5);
		elseif strcmp(solver{s},'prsvm+')
			semilogx(sec(s,2:counter(s)),accuracy(s,2:counter(s)),'k+:','LineWidth',5);
		elseif strcmp(solver{s},'treeranksvm')
			semilogx(sec(s,2:counter(s)),accuracy(s,2:counter(s)),'r--','LineWidth',5);
		end
		if s == 1
			hold on;
		end
	end
	for s=1:length(solver)
		if success(s) == 0
			continue;
		end
		if strcmp(solver{s},'tree-tron')
			semilogx(get(gca,'xlim'),[accuracy(s,stop(s)) accuracy(s,stop(s))],'c*-','LineWidth',3);
		end
	end
	le=legend(solver,'Location','SouthEast');
	set (le,'FontSize',24);
	xlabel('Training Time (Seconds)','FontSize',30);
	ylabel('Pairwise Accuracy','FontSize',30);
	print(h,'-dpng',['fig5_6/',inputlist{i},'.accuracy.png']);
	clear h;
	clear time line t act y counter fun accuracy opt l1opt opttemp sec;
end
