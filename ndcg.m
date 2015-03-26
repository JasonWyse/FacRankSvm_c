inputlist = {'MQ2007','MQ2008','MSLR','YAHOO_SET1','YAHOO_SET2'};
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
		fid = fopen(['log/',inputlist{i},'.',solver{s},'.ndcg.log'],'r');
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
					if (touched(s) == 0 && strcmp(t{j},'|g|'))
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
			elseif (strcmp(inputlist{i},'YAHOO_SET1') || strcmp(inputlist{i},'YAHOO_SET2'))
				if (length(t) > 1 && strcmp(t{2},'(YAHOO)'))
					ndcg_value(s,counter(s)) = str2double(t{4}(1:length(t{4})-1));
				end
			elseif (length(t) > 1 && strcmp(t{2},'(LETOR)'))
				ndcg_value(s,counter(s)) = str2double(t{4}(1:length(t{4})-1));
			end
			line =  fgetl(fid);
		end
		if (counter(s)==1)
			break;
		end
		fclose(fid);
		time = max(time,sec(s,counter(s)));
	end
	h = figure;
	set(gca,'fontsize',14);
	for s = 1:length(solver)
		if success(s) == 0
			continue;
		end
		if (sec(s,counter(s))<time)
			counter(s) = counter(s)+1;
			sec(s,counter(s))=time;
			ndcg_value(s,counter(s)) = ndcg_value(s,counter(s)-1);
		end
	end
	h = figure;
	set(gca,'fontsize',14);
	for s=1:length(solver)
		if success(s) == 0
			continue;
		end
		if strcmp(solver{s},'tree-tron')
			semilogx(sec(s,2:counter(s)),ndcg_value(s,2:counter(s)),'c-','LineWidth',5);
		elseif strcmp(solver{s},'prsvm+')
			semilogx(sec(s,2:counter(s)),ndcg_value(s,2:counter(s)),'k+:','LineWidth',5);
		else
			semilogx(sec(s,2:counter(s)),ndcg_value(s,2:counter(s)),'r--','LineWidth',5);
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
			semilogx(get(gca,'xlim'),[ndcg(s,stop(s)) ndcg(s,stop(s))],'c*-','LineWidth',3);
		end
	end
	le=legend(solver,'Location','SouthEast');
	set (le,'FontSize',18);
	xlabel('Training Time (Seconds)','FontSize',18);
	ylabel('Pairwise Accuracy','FontSize',18);
	print(h,'-dpng',['fig5_6/',inputlist{i},'.ndcg.png']);
	clear h;
	clear time line t act y counter fun acc opt l1opt opttemp sec;
end
