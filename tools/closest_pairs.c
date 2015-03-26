#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */  
	int *query;
};
void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

long int elements;
long long pairs;
static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}
void read_problem(const char *filename);
void generate_rank(void);
void output_target(double value);
void output(int index, double value);

struct feature_node *x_space;
struct problem prob;

int main(int argc, char **argv)
{
	if (argc != 2 && argc != 4)
	{
		fprintf(stderr,"Usage: %s file_to_generate_pair_data\n",argv[0]);
		return 0;
	}
	char input_file_name[1024];
	strcpy(input_file_name,argv[argc-1]);
	read_problem(input_file_name);
	generate_rank();
	free(prob.y);
	free(prob.x);
	free(prob.query);
	free(x_space);
	free(line);
	return 0;
}

void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		prob.l++;
	}
	rewind(fp);

	prob.bias=-1;
	prob.query = Malloc(int,prob.l);
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		prob.query[i] = 0;
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			if (!strcmp(idx,"qid"))
			{
				errno = 0;
				prob.query[i] = (int) strtol(val, &endptr,10);
				if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);
			}
			else
			{

				errno = 0;
				x_space[j].index = (int) strtol(idx,&endptr,10);
				if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
					exit_input_error(i+1);
				else
					inst_max_index = x_space[j].index;

				errno = 0;
				x_space[j].value = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);

				++j;
			}
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		x_space[j++].index = -1;
	}
	prob.n=max_index;

	fclose(fp);
	
}

void generate_rank()
{
	pairs=0;
	int i,j;
	for (i=0; i<prob.l-1;i++)
	{
		for (j=i+1;j<prob.l;j++)
		{
			int tmp = int(prob.y[i] - prob.y[j]);
			if (tmp*tmp == 1 && prob.query[i] == prob.query[j])
			{
				output_target(1);
				pairs++;
				struct feature_node *s=prob.x[i];
				struct feature_node *t=prob.x[j];
				int i_coeff;
				int j_coeff;
				if (prob.y[i]>prob.y[j])
				{
					i_coeff=1;
					j_coeff=-1;
				}
				else
				{
					i_coeff=-1;
					j_coeff=1;
				}

				while(s->index!=-1 || t->index!=-1)
				{
					if (s->index == -1)
					{
						while (t->index !=-1)
						{
							output(t->index,t->value*j_coeff);
							t++;
						}
						break;
					}
					if (t->index == -1)
					{
						while (s->index !=-1)
						{
							output(s->index,s->value*i_coeff);
							s++;
						}
						break;
					}
					
					if (s->index==t->index)
					{
						output(s->index,s->value*i_coeff+t->value*j_coeff);
						s++;
						t++;
					}

					else if(s->index<t->index)
					{
						output(s->index,s->value*i_coeff);
						s++;
					}
					else
					{
						output(t->index,t->value*j_coeff);
						t++;
					}
				}			
				printf("\n");
			}
		}
	}
}


void output_target(double value)
{
	printf("%g ",value);
}

void output(int index, double value)
{
	if (value!=0)
		printf("%d:%g ",index, value);
}
