#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include "selectiontree.h"
struct id_and_value
{
	int id;
	double value;
};

struct feature_node
{
	int index;
	double value;
};

int compare_id_and_value(const void *a, const void *b)
{
	struct id_and_value *ia = (struct id_and_value *)a;
	struct id_and_value *ib = (struct id_and_value *)b;
	if(ia->value > ib->value)
		return 1;
	if(ia->value < ib->value)
		return -1;
	return 0;
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#ifndef min
template <class T> static inline T min(T x,T y) { return (x>y)?y:x; }
#endif
void group_queries(const int *query_id, int l, int *nr_query_ret, int **start_ret, int **count_ret, int *perm)
{
	int max_nr_query = 16;
	int nr_query = 0;
	int *query = Malloc(int,max_nr_query);
	int *count = Malloc(int,max_nr_query);
	int *data_query = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_query = (int)query_id[i];
		int j;
		for(j=0;j<nr_query;j++)
		{
			if(this_query == query[j])
			{
				++count[j];
				break;
			}
		}
		data_query[i] = j;
		if(j == nr_query)
		{
			if(nr_query == max_nr_query)
			{
				max_nr_query *= 2;
				query = (int *)realloc(query,max_nr_query * sizeof(int));
				count = (int *)realloc(count,max_nr_query * sizeof(int));
			}
			query[nr_query] = this_query;
			count[nr_query] = 1;
			++nr_query;
		}
	}

	int *start = Malloc(int,nr_query);
	start[0] = 0;
	for(i=1;i<nr_query;i++)
		start[i] = start[i-1] + count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_query[i]]] = i;
		++start[data_query[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_query;i++)
		start[i] = start[i-1] + count[i-1];

	*nr_query_ret = nr_query;
	*start_ret = start;
	*count_ret = count;
	free(query);
	free(data_query);
}

static char *line = NULL;
static int max_line_len;

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

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

void eval(double *label, double *target, int *query, int l)
{
	int q,i,j,k;
	int nr_query;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int, l);
	id_and_value *pi;
	int true_query;
	long long totalnc = 0, totalnd = 0;
	long long nc = 0;
	long long nd = 0;
	double tmp;
	double accuracy = 0;
	int *l_plus;
	int *int_y;
	int same_y = 0;
	double *idcg;
	double *dcg;
	double idcg_yahoo,dcg_yahoo,size_yahoo;
	double ndcg_yahoo = 0;
	double meanndcg = 0;
	double tmp_ndcg;
	long long pairs = 0;
	selectiontree *T;
	group_queries(query, l, &nr_query, &start, &count, perm);
	true_query = nr_query;
	for (q=0;q<nr_query;q++)
	{
		//pairwise accuracy
		nc = 0;
		nd = 0;
		l_plus = new int[count[q]];
		int_y = new int[count[q]];
		pi = new id_and_value[count[q]];
		for (i=0;i<count[q];i++)
		{
			pi[i].id = i;
			pi[i].value = label[perm[i + start[q]]];
		}
		qsort(pi, count[q], sizeof(id_and_value), compare_id_and_value);
		int_y[pi[0].id] = 1;
		same_y = 0;
		k = 1;
		for(i=1;i<count[q];i++)
		{
			if (pi[i-1].value < pi[i].value)
			{
				same_y = 0;
				k++;
			}
			else
				same_y++;
			int_y[pi[i].id] = k;
			nc += (i - same_y);
		}
		pairs += nc;
		for (i=0;i<count[q];i++)
		{
			pi[i].id = i;
			pi[i].value = target[perm[i + start[q]]];
		}
		qsort(pi, count[q], sizeof(id_and_value), compare_id_and_value);
		//total pairs
		T = new selectiontree(k);
		j = 0;
		for (i=0;i<count[q];i++)
		{
			while (j<count[q] && ( pi[j].value < pi[i].value))
			{
				T->insert_node(int_y[pi[j].id], tmp);
				j++;
			}
			T->larger(int_y[pi[i].id], &l_plus[pi[i].id], &tmp);
		}
		delete T;

		for (i=0;i<count[q];i++)
			nd += l_plus[i];
		nc -= nd;
		if (nc != 0 || nd != 0)
			accuracy += double(nc)/double(nc+nd);
		else
			true_query--;
		totalnc += nc;
		totalnd += nd;
		delete[] l_plus;
		delete[] int_y;
		delete[] pi;
	}
	printf("Total pairs = %lld\n",pairs);
	printf("Pairwise accuracy = %g%% (ranking)\n",(double)totalnc/(double)(totalnc+totalnd)*100);
	for (q=0;q<nr_query;q++)
	{
		//mean ndcg
		idcg = new double[count[q]];
		dcg = new double[count[q]];
		size_yahoo = min(count[q],10);
		tmp_ndcg = 0;
		pi = new id_and_value[count[q]];
		for (i=0;i<count[q];i++)
		{
			pi[i].id = perm[i + start[q]];
			pi[i].value = label[perm[i + start[q]]];
		}
		qsort(pi, count[q], sizeof(id_and_value), compare_id_and_value);
		idcg_yahoo = 0;
		dcg_yahoo = 0;
		idcg[0] = pow(2.0,pi[count[q]-1].value) - 1;
		for (i=1;i<count[q];i++)
			idcg[i] = idcg[i-1] + (pow(2.0,pi[count[q]-1 - i].value) - 1) * log(2.0) / log(i+1.0);
		for (i=0;i<size_yahoo;i++)
			idcg_yahoo += (pow(2.0,pi[count[q]-1 - i].value) - 1) * log(2.0) / log(i+2.0);
		for (i=0;i<count[q];i++)
		{
			pi[i].id = perm[i + start[q]];
			pi[i].value = target[perm[i + start[q]]];
		}
		qsort(pi, count[q], sizeof(id_and_value), compare_id_and_value);
		dcg[0] = pow(2.0, label[pi[count[q] - 1].id]) - 1;
		for (i=1;i<count[q];i++)
			dcg[i] = dcg[i-1] + (pow(2.0, label[pi[count[q] - 1 - i].id]) - 1) * log(2.0) / log(i + 1.0);
		for (i=0;i<size_yahoo;i++)
			dcg_yahoo += (pow(2.0,label[pi[count[q]-1 - i].id]) - 1) * log(2.0) / log(i+2.0);
		if (idcg[0]>0)
			for (i=0;i<count[q];i++)
				tmp_ndcg += dcg[i]/idcg[i];
		else
			tmp_ndcg = 0;
		meanndcg += tmp_ndcg/count[q];

		if (idcg_yahoo > 0)
			ndcg_yahoo += dcg_yahoo / idcg_yahoo;
		else
			ndcg_yahoo += 1;
		delete[] pi;
		delete[] idcg;
		delete[] dcg;
	}
	meanndcg /= nr_query;
	ndcg_yahoo /= nr_query;
	printf("MeanNDCG (LETOR) = %g (ranking)\n", meanndcg);
	printf("NDCG (YAHOO) = %g (ranking)\n", ndcg_yahoo);
	free(start);
	free(count);
	free(perm);
}

void do_predict(FILE *input, FILE *output)
{
	int total = 0;
	int i;
	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
		total++;
	rewind(input);
	double *dvec_t = new double[total];
	double *ivec_t = new double[total];
	int *query = new int[total];
	total = 0;
	while(readline(input) != NULL)
	{
		i = 0;
		char* label,*endptr;
		label = strtok(line," \t\n");
		dvec_t[total] = strtod(label,&endptr);
		total++;
	}
	total = 0;
	while(readline(output) != NULL)
	{
		i = 0;
		char *idx, *val, *label, *endptr;

		label = strtok(line," \t");
		ivec_t[total] = strtod(label,&endptr);
		query[total] = 0;
		if(endptr == label)
			exit_input_error(total+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			if (!strcmp(idx,"qid"))
			{
				errno = 0;
				query[total] = (int) strtol(val, &endptr,10);
				if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);
			}
		}
		total++;
	}
	eval(ivec_t,dvec_t,query,total);
}

void exit_with_help()
{
	printf(
			"Usage: predict [options] test_file model_file output_file\n"
			"options:\n"
			"-q : quiet mode (no outputs)\n"
			"-p : whether to use pair-wise evaluation criteria (Kendall's tau and pair-wise accuracy), 0 or 1 (default 1)\n"
			"-N n: Compute NDCG@1 to NDCG@n, if 0 then only meanNDCG is computed, if -1 then NDCG is disabled (default 10)\n"  
		  );
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	if (argc<3)
	{
		fprintf(stderr,"Usage: %s outputed_prediction test_file\n",argv[0]);
		return 0;
	}
	// parse options
	input = fopen(argv[1],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[1]);
		exit(1);
	}

	output = fopen(argv[2],"r");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[2]);
		exit(1);
	}

	do_predict(input, output);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}
