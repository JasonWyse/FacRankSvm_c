inputlist = {'MQ2007-c100','MQ2007-c1','MQ2007-c0.0001','MQ2008-c100','MQ2008-c1','MQ2008-c0.0001','YAHOO_SET2-c100','YAHOO_SET2-c1','YAHOO_SET2-c0.0001','MQ2007-list-c100','MQ2007-list-c1','MQ2007-list-c0.0001','MQ2008-list-c100','MQ2008-list-c1','MQ2008-list-c0.0001'};
solver = {'tree-tron','prsvm+','treeranksvm'};

eta0=1e-4;
for i = 1:length(inputlist)
		time = 0;
		for s = 1:length(solver)
			counter(s)=1;
			sec(s,1)=0;
			fid = fopen(['log/',inputlist{i},'.',solver{s},'.log'],'r');
			if (fid == -1)
				break;
			end
			line = fgetl(fid);
			while(ischar(line))
				t = strread(line,'%s','delimiter',' \t');
				if (strcmp(t{1},'iter'))
					counter(s)=counter(s)+1;
					for j = 2:length(t)
						if (strcmp(t{j},'act'))
							act = str2double(t{j+1});
						elseif (strcmp(t{j},'pre'))
							pre = str2double(t{j+1});
						elseif (strcmp(t{j},'f'))
							fun(s,counter(s)-1)=str2double(t{j+1});
						end
					end
				elseif (strcmp(t{1},'Time'))
					sec(s,counter(s))=str2double(t{2});
				elseif (length(t) >= 4 && strcmp(t{3},'TIME'))
					sec(s,counter(s)+1)=str2double(t{4});
				end
				line =  fgetl(fid);
			end
			if (counter(s)==1)
				break;
			end
			fun(s,counter(s)) = fun(s,counter(s)-1);
			if (s==3 || act>=eta0*pre)
				fun(s,counter(s)) = fun(s,counter(s)) - act;
			end
			opttemp(s) = min(fun(s,1:counter(s)));
			time = max(time,sec(s,counter(s)));
			fclose(fid);
		end
	
		opt = min(opttemp(1:2));
		y = (fun-opt)/opt;
		l1opt = opttemp(3);
		y1 = (fun(3,:)-l1opt)/l1opt;
		h = figure;
		set(gca,'fontsize',16);
		loglog(sec(1,1:counter(1)),y(1,1:counter(1)),'c-','LineWidth',15);
		hold on;
		loglog(sec(2,1:counter(2)),y(2,1:counter(2)),'k+:','LineWidth',7);
		loglog(sec(3,1:counter(3)),y1(1:counter(3)),'r--','LineWidth',7);
		xlabel('Training Time (Seconds)','FontSize',30);
		ylabel('Difference to Opt. Fun. Value','FontSize',30);
		inputlist{i}
		le=legend(solver);
		set (le,'FontSize',16);
		print(h,'-dpng',['fig7/',inputlist{i},'.png']);
		clear h;

		clear time line t act y counter fun acc ndcg opt l1opt opttemp sec;
end
