#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

void exit_with_help()
{
	printf(
	"Usage: svm-scale [options] data_filename\n"
	"options:\n"
	"-l lower : x scaling lower limit (default -1)\n"
	"-u upper : x scaling upper limit (default +1)\n"
	"-y y_lower y_upper : y scaling limits (default: no y scaling)\n"
	"-s save_filename : save scaling parameters to save_filename\n"
	"-r restore_filename : restore scaling parameters from restore_filename\n"
	);
	exit(1);
}

char *line = NULL;
int max_line_len = 1024;
double lower=-1.0,upper=1.0,y_lower,y_upper;
int y_scaling = 0;
double *feature_max;
double *feature_min;
int *appear_count;
double y_max = -DBL_MAX;
double y_min = DBL_MAX;
int max_index;
int min_index;
long int num_nonzeros = 0;
long int new_num_nonzeros = 0;

#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))

void output_target(double value);
void output(int index, double value);
char* readline(FILE *input);

int main(int argc,char **argv)
{
	int i,index;
	int l = 0;
	FILE *fp, *fp_restore = NULL;
	char *save_filename = NULL;
	char *restore_filename = NULL;
	char *idx, *val, *label;
	char *endptr;
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'l': lower = atof(argv[i]); break;
			case 'u': upper = atof(argv[i]); break;
			case 'y':
				y_lower = atof(argv[i]);
				++i;
				y_upper = atof(argv[i]);
				y_scaling = 1;
				break;
			case 's': save_filename = argv[i]; break;
			case 'r': restore_filename = argv[i]; break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}

	if(!(upper > lower) || (y_scaling && !(y_upper > y_lower)))
	{
		fprintf(stderr,"inconsistent lower/upper specification\n");
		exit(1);
	}

	if(restore_filename && save_filename)
	{
		fprintf(stderr,"cannot use -r and -s simultaneously\n");
		exit(1);
	}

	if(argc != i+1)
		exit_with_help();

	fp=fopen(argv[i],"r");

	if(fp==NULL)
	{
		fprintf(stderr,"can't open file %s\n", argv[i]);
		exit(1);
	}

	line = (char *) malloc(max_line_len*sizeof(char));

	/* assumption: min index of attributes is 1 */
	/* pass 1: find out max index of attributes */
	max_index = 0;
	min_index = 1;
	if(restore_filename)
	{
		int idx, c;

		fp_restore = fopen(restore_filename,"r");
		if(fp_restore==NULL)
		{
			fprintf(stderr,"can't open file %s\n", restore_filename);
			exit(1);
		}

		c = fgetc(fp_restore);
		if(c == 'y')
		{
			readline(fp_restore);
			readline(fp_restore);
			readline(fp_restore);
		}
		readline(fp_restore);
		readline(fp_restore);

		while(fscanf(fp_restore,"%d %*f %*f\n",&idx) == 1)
			max_index = max(idx,max_index);
		rewind(fp_restore);
	}
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label
		l++;
		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			index = (int) strtol(p,&endptr,10);
			if (endptr == p)
				continue;
			max_index = max(max_index, index);
			min_index = min(min_index, index);
			++num_nonzeros;
		}
	}
	if(min_index < 1)
		fprintf(stderr,
				"WARNING: minimal feature index is %d, but indices should start from 1\n", min_index);

	rewind(fp);

	feature_max = (double *)malloc((max_index+1)* sizeof(double));
	feature_min = (double *)malloc((max_index+1)* sizeof(double));
	appear_count = (int *)malloc((max_index+1)* sizeof(int));

	if(feature_max == NULL || feature_min == NULL)
	{
		fprintf(stderr,"can't allocate enough memory\n");
		exit(1);
	}

	for(i=0;i<=max_index;i++)
	{
		feature_max[i]=-DBL_MAX;
		feature_min[i]=DBL_MAX;
		appear_count[i]=0;
	}

	/* pass 2: find out min/max value */
	while(readline(fp)!=NULL)
	{
		double target;
		double value;
		label = strtok(line," \t"); // label
		target = strtod(label,&endptr);
		y_max = max(y_max,target);
		y_min = min(y_min,target);
		// features
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			if (strcmp(idx,"qid"))
			{
				index = (int) strtol(idx,&endptr,10);
				appear_count[index]++;

				value = strtod(val,&endptr);
				feature_max[index]=max(feature_max[index],value);
				feature_min[index]=min(feature_min[index],value);
			}
		}
	}
	for (i=0;i<=max_index;i++)
		if (appear_count[i] < l)
		{
			feature_max[i] = max(feature_max[i],0);
			feature_min[i] = min(feature_min[i],0);
		}
	
	rewind(fp);

	/* pass 2.5: save/restore feature_min/feature_max */

	if(restore_filename)
	{
		/* fp_restore rewinded in finding max_index */
		int idx, c;
		double fmin, fmax;
		int next_index = 1;
		if((c = fgetc(fp_restore)) == 'y')
		{
			fscanf(fp_restore, "%lf %lf\n", &y_lower, &y_upper);
			fscanf(fp_restore, "%lf %lf\n", &y_min, &y_max);
			y_scaling = 1;
		}
		else
			ungetc(c, fp_restore);

		if (fgetc(fp_restore) == 'x')
		{
			fscanf(fp_restore, "%lf %lf\n", &lower, &upper);
			while(fscanf(fp_restore,"%d %lf %lf\n",&idx,&fmin,&fmax)==3)
			{
				feature_min[idx] = fmin;
				feature_max[idx] = fmax;

				next_index = idx + 1;
			}
		}
		fclose(fp_restore);
	}

	if(save_filename)
	{
		FILE *fp_save = fopen(save_filename,"w");
		if(fp_save==NULL)
		{
			fprintf(stderr,"can't open file %s\n", save_filename);
			exit(1);
		}
		if(y_scaling)
		{
			fprintf(fp_save, "y\n");
			fprintf(fp_save, "%.16g %.16g\n", y_lower, y_upper);
			fprintf(fp_save, "%.16g %.16g\n", y_min, y_max);
		}
		fprintf(fp_save, "x\n");
		fprintf(fp_save, "%.16g %.16g\n", lower, upper);
		for(i=1;i<=max_index;i++)
		{
			if(feature_min[i]!=feature_max[i])
				fprintf(fp_save,"%d %.16g %.16g\n",i,feature_min[i],feature_max[i]);
		}
		if(min_index < 1)
			fprintf(stderr,
					"WARNING: scaling factors with indices smaller than 1 are not stored to the file %s.\n", save_filename);

		fclose(fp_save);
	}

	/* pass 3: scale */
	while(readline(fp)!=NULL)
	{
		int next_index=1;
		double target;
		double value;

		label = strtok(line," \t"); // label
		target = strtod(label,&endptr);
		output_target(target);
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			if (strcmp(idx,"qid"))
			{
				index = (int) strtol(idx,&endptr,10);
				value = strtod(val,&endptr);

				for(i=next_index;i<index;i++)
					output(i,0);

				output(index,value);
				next_index=index+1;
			}
			else
				printf("%s:%s ",idx,val);
		}
		for(i=next_index;i<=max_index;i++)
			output(i,0);
		printf("\n");
	}

	if (new_num_nonzeros > num_nonzeros)
		fprintf(stderr,
				"WARNING: original #nonzeros %ld\n"
				"         new      #nonzeros %ld\n"
				"Use -l 0 if many original feature values are zeros\n",
				num_nonzeros, new_num_nonzeros);

	free(line);
	free(feature_max);
	free(feature_min);
	fclose(fp);
	return 0;
}

char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void output_target(double value)
{
	if(y_scaling)
	{
		if(value == y_min)
			value = y_lower;
		else if(value == y_max)
			value = y_upper;
		else value = y_lower + (y_upper-y_lower) *
			(value - y_min)/(y_max-y_min);
	}
	printf("%g ",value);
}

void output(int index, double value)
{
	/* skip single-valued attribute */
	if(feature_max[index] == feature_min[index])
		return;

	if(value == feature_min[index])
		value = lower;
	else if(value == feature_max[index])
		value = upper;
	else
		value = lower + (upper-lower) *
			(value-feature_min[index])/
			(feature_max[index]-feature_min[index]);

	if(value != 0)
	{
		printf("%d:%g ",index, value);
		new_num_nonzeros++;
	}
}
