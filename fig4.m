inputlist = {'MQ2007','MQ2008'};
solver = {'y-avltree'};
for i = 1:length(inputlist)
	grad0=0;
	stop=zeros(length(solver),1);
	clear time counter sec line fun y opt opttemp;
	counter = zeros(length(solver),1);
	time = 0;
	for s = 1:length(solver)
		counter(s)=1;
		fid = fopen(['log/',inputlist{i},'.',solver{s},'.fig4.log'],'r');
		line = fgetl(fid);
		while(ischar(line))
			t = strread(line,'%s','delimiter',' \t');
			if (strcmp(t{1},'FUN'))
				fun(s,counter(s))=str2double(t{2});
				sec(s,counter(s))=str2double(t{4});
				counter(s)=counter(s)+1;
			elseif (length(t)>=12 && strcmp(t{11},'|g|'))
				grad = str2double(t{12});
				if (grad0 == 0)
					grad0 = grad;
				elseif (grad/grad0 <= 0.001)
					stop(s) = max(stop(s),fun(s,counter(s)-1));
				end
			end
			line =  fgetl(fid);
		end
		counter(s) = counter(s) - 1;
		opttemp(s) = min(fun(s,1:counter(s)));
		time = max(time,sec(s,counter(s)));
		fclose(fid);
	end
	opt = min(opttemp);
	y = (fun-opt)/abs(opt);
	stop=(stop-opt)/abs(opt);
	h = figure;
	set(gca,'fontsize',24);
	for s = 1:length(solver)
		if strcmp(solver{s},'direct-count')
			semilogy(sec(s,1:counter(s)),y(s,1:counter(s)),'r^-','LineWidth',2);
		elseif strcmp(solver{s},'y-rbtree')
			semilogy(sec(s,1:counter(s)),y(s,1:counter(s)),'-.k*','LineWidth',1);
		elseif strcmp(solver{s},'wx-rbtree')
			semilogy(sec(s,1:counter(s)),y(s,1:counter(s)),'cs--','LineWidth',1);
		elseif strcmp(solver{s},'selectiontree')
			semilogy(sec(s,1:counter(s)),y(s,1:counter(s)),'bx--','LineWidth',1);
		elseif strcmp(solver{s},'y-avltree')
			semilogy(sec(s,1:counter(s)),y(s,1:counter(s)),'go-','LineWidth',1);
		elseif strcmp(solver{s},'y-aatree')
			semilogy(sec(s,1:counter(s)),y(s,1:counter(s)),'md-.','LineWidth',1);
		end
		if (s == 1)
			hold on;
		end
	end
	if strcmp(inputlist{i},'MQ2007-list') || strcmp(inputlist{i},'MQ2008-list')
		axis([0 500 0 1]);
	end
	xlabel('Training Time (Seconds)','FontSize',26);
	ylabel('Difference to Opt. Fun. Value','FontSize',30);
	le=legend(solver,'Location','SouthWest');
	set (le,'FontSize',24);
	semilogy(get(gca,'xlim'),[stop(1) stop(1)],'--','Color',[0.5,0.5,0.5]);
	print(h,'-dpng',['fig4/',inputlist{i},'.fig4.png']);
	clear h;
end
