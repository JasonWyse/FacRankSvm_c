#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <time.h>
#include "linear.h"
#include "tron.h"
#include "binarytrees.h"
#include "selectiontree.h"

#ifdef FIGURE56
struct feature_node *x_spacetest;
struct problem probtest;
#endif
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
#ifdef FIGURE56
void evaluate_test(double* w)
{
	int i;
	double *true_labels = Malloc(double,probtest.l);
	double *dec_values = Malloc(double,probtest.l);
	if(&probtest != NULL)
	{
		for(i = 0; i < probtest.l; ++i)
		{
			feature_node *x = probtest.x[i];
			true_labels[i] = probtest.y[i];
			double predict_label = 0;
			for(; x->index != -1; ++x)
				predict_label += w[x->index-1]*x->value;
			dec_values[i] = predict_label;
		}
	}
	double result[3];
	eval_list(true_labels, dec_values, probtest.query, probtest.l, result);
	info("Pairwise Accuracy = %g%%\n",result[0]*100);
	info("MeanNDCG (LETOR) = %g\n",result[1]);
	info("NDCG (YAHOO) = %g\n",result[2]);
	free(true_labels);
	free(dec_values);
}
#endif

class l2r_l2_svc_fun: public function
{
	public:
		l2r_l2_svc_fun(const problem *prob, double *C);
		~l2r_l2_svc_fun();

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *s, double *Hs);

		int get_nr_variable(void);

	protected:
		void Xv(double *v, double *Xv);
		void subXv(double *v, double *Xv);
		void subXTv(double *v, double *XTv);

		double *C;
		double *z;
		double *D;
		int *I;
		int sizeI;
		const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	I = new int[l];
	this->C = C;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
	public:
		l2r_l2_svr_fun(const problem *prob, double *C, double p);

		double fun(double *w);
		void grad(double *w, double *g);

	private:
		double p;
};

l2r_l2_svr_fun::l2r_l2_svr_fun(const problem *prob, double *C, double p):
	l2r_l2_svc_fun(prob, C)
{
	this->p = p;
}

double l2r_l2_svr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];
		if(d < -p)
			f += C[i]*(d+p)*(d+p);
		else if(d > p)
			f += C[i]*(d-p)*(d-p);
	}

	return(f);
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	sizeI = 0;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];

		// generate index set I
		if(d < -p)
		{
			z[sizeI] = C[i]*(d+p);
			I[sizeI] = i;
			sizeI++;
		}
		else if(d > p)
		{
			z[sizeI] = C[i]*(d-p);
			I[sizeI] = i;
			sizeI++;
		}

	}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

static int compare_id_and_value(const void *a, const void *b)
{
	struct id_and_value *ia = (struct id_and_value *)a;
	struct id_and_value *ib = (struct id_and_value *)b;
	if(ia->value > ib->value)
		return -1;
	if(ia->value < ib->value)
		return 1;
	return 0;
}

class y_rbtree_rank_fun: public function
{
	public:
		y_rbtree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count);
		~y_rbtree_rank_fun();

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *s, double *Hs);

		int get_nr_variable(void);

	protected:
		void Xv(double *v, double *Xv);
		void XTv(double *v, double *XTv);

		double C;
		double *z;
		int *l_plus;
		int *l_minus;
		double *alpha_plus;
		double *alpha_minus;
		const problem *prob;
		int nr_subset;
		int *perm;
		int *start;
		int *count;
		id_and_value **pi;
};

y_rbtree_rank_fun::y_rbtree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count)
{
	int l=prob->l;
	this->prob = prob;
	this->nr_subset = nr_subset;
	this->perm = perm;
	this->start = start;
	this->count = count;
	this->C = C;
	l_plus = new int[l];
	l_minus = new int[l];
	alpha_plus = new double[l];
	alpha_minus = new double[l];
	z = new double[l];
	pi = new id_and_value* [nr_subset];
	for (int i=0;i<nr_subset;i++)
		pi[i] = new id_and_value[count[i]];
}

y_rbtree_rank_fun::~y_rbtree_rank_fun()
{
	delete[] l_plus;
	delete[] l_minus;
	delete[] alpha_plus;
	delete[] alpha_minus;
	delete[] z;
	for (int i=0;i<nr_subset;i++)
		delete[] pi[i];
	delete[] pi;
}

double y_rbtree_rank_fun::fun(double *w)
{
	int i,j,k;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	rbtree *T;
	Xv(w,z);
	for (i=0;i<nr_subset;i++)
	{
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);
		T=new rbtree(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k++;
			}
			T->count_smaller(y[pi[i][j].id],&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new rbtree(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(y[pi[i][j].id],&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	return(f);
}

void y_rbtree_rank_fun::grad(double *w, double *g)
{
	int i;
	int l=prob->l;
	double *ATAXw;
	ATAXw = new double[l];
	int w_size=get_nr_variable();
	for (i=0;i<l;i++)
		ATAXw[i]=(double)l_plus[i]-(double)l_minus[i]+((double)l_plus[i]+(double)l_minus[i])*z[i]-alpha_plus[i]-alpha_minus[i];
	XTv(ATAXw, g);
	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*C*g[i];
	delete[] ATAXw;
}

int y_rbtree_rank_fun::get_nr_variable(void)
{
	return prob->n;
}

void y_rbtree_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *y=prob->y;
	double *wa = new double[l];
	rbtree *T;
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		T=new rbtree(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k++;
			}
			alpha_plus_minus[pi[i][j].id]=T->vector_sum_smaller(y[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new rbtree(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k--;
			}
			alpha_plus_minus[pi[i][j].id]+=T->vector_sum_larger(y[pi[i][j].id]);
		}
		delete T;
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}

void y_rbtree_rank_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void y_rbtree_rank_fun::XTv(double *v, double *XTv)
{
	int i;
	int l = prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class y_avltree_rank_fun: public y_rbtree_rank_fun
{
	public:
		y_avltree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count): y_rbtree_rank_fun(prob, C, nr_subset, perm, start, count){};
		double fun(double *w);
		void Hv(double *s, double *Hs);
};

double y_avltree_rank_fun::fun(double *w)
{
	int i,j,k;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	avl *T;
	Xv(w,z);
	for (i=0;i<nr_subset;i++)
	{
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);
		T=new avl(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k++;
			}
			T->count_smaller(y[pi[i][j].id],&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new avl(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(y[pi[i][j].id],&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	return(f);
}

void y_avltree_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *y=prob->y;
	double *wa = new double[l];
	avl *T;
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		T=new avl(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k++;
			}
			alpha_plus_minus[pi[i][j].id]=T->vector_sum_smaller(y[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new avl(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k--;
			}
			alpha_plus_minus[pi[i][j].id]+=T->vector_sum_larger(y[pi[i][j].id]);
		}
		delete T;
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}

class y_aatree_rank_fun: public y_rbtree_rank_fun
{
	public:
		y_aatree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count): y_rbtree_rank_fun(prob, C, nr_subset, perm, start, count){};
		double fun(double *w);
		void Hv(double *s, double *Hs);
};

double y_aatree_rank_fun::fun(double *w)
{
	int i,j,k;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	aatree *T;
	Xv(w,z);
	for (i=0;i<nr_subset;i++)
	{
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);
		T=new aatree(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k++;
			}
			T->count_smaller(y[pi[i][j].id],&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new aatree(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(y[pi[i][j].id],&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	return(f);
}

void y_aatree_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *y=prob->y;
	double *wa = new double[l];
	aatree *T;
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		T=new aatree(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k++;
			}
			alpha_plus_minus[pi[i][j].id]=T->vector_sum_smaller(y[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new aatree(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k--;
			}
			alpha_plus_minus[pi[i][j].id]+=T->vector_sum_larger(y[pi[i][j].id]);
		}
		delete T;
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}
class wx_rbtree_rank_fun: public y_rbtree_rank_fun
{
	public:
		wx_rbtree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count);
		double fun(double *w);
		void Hv(double *s, double *Hs);
};

wx_rbtree_rank_fun::wx_rbtree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count): y_rbtree_rank_fun(prob, C, nr_subset, perm, start, count)
{
	int i,j;
	double *y = prob->y;
	for (i=0;i<nr_subset;i++)
	{
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = y[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);
	}
}

double wx_rbtree_rank_fun::fun(double *w)
{
	int i,j,k;
	double f=0;
	int l=prob->l;
	int w_size=get_nr_variable();
	rbtree *T;
	Xv(w,z);
	for (i=0;i<nr_subset;i++)
	{
		T=new rbtree(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(pi[i][j].value<pi[i][k].value))
			{
				T->insert_node(z[pi[i][k].id],z[pi[i][k].id]);
				k++;
			}
			T->count_smaller(z[pi[i][j].id]+1.0,&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;

		k = count[i]-1;
		T = new rbtree(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(pi[i][j].value>pi[i][k].value))
			{
				T->insert_node(z[pi[i][k].id],z[pi[i][k].id]);
				k--;
			}
			T->count_larger(z[pi[i][j].id]-1.0,&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	return(f);
}

void wx_rbtree_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *wa = new double[l];
	rbtree *T;
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		T=new rbtree(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(pi[i][j].value<pi[i][k].value))
			{
				T->insert_node(z[pi[i][k].id],wa[pi[i][k].id]);
				k++;
			}
			alpha_plus_minus[pi[i][j].id]=T->vector_sum_smaller(z[pi[i][j].id]+1);
		}
		delete T;
		k = count[i]-1;
		T = new rbtree(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(pi[i][j].value>pi[i][k].value))
			{
				T->insert_node(z[pi[i][k].id],wa[pi[i][k].id]);
				k--;
			}
			alpha_plus_minus[pi[i][j].id]+=T->vector_sum_larger(z[pi[i][j].id]-1);
		}
		delete T;
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}

class selection_rank_fun: public y_rbtree_rank_fun
{
	public:
		selection_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count);
		~selection_rank_fun();

		double fun(double *w);
		void Hv(double *s, double *Hs);
	protected:
		int *int_y;
		int *nr_class;

};
selection_rank_fun::selection_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count): y_rbtree_rank_fun(prob, C, nr_subset, perm, start, count)
{
	int i,j,k;
	double *y=prob->y;
	int_y = new int[prob->l];
	nr_class = new int[nr_subset];
	for (i=0;i<nr_subset;i++)
	{
		k=1;
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id=perm[j+start[i]];
			pi[i][j].value=y[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);

		int_y[pi[i][count[i]-1].id]=1;
		for(j=count[i]-2;j>=0;j--)
		{
			if (pi[i][j].value>pi[i][j+1].value)
				k++;
			int_y[pi[i][j].id]=k;
		}
		nr_class[i]=k;
	}
}

selection_rank_fun::~selection_rank_fun()
{
	delete[] int_y;
	delete[] nr_class;
}

double selection_rank_fun::fun(double *w)
{
	int i,j,k;
	double f=0;
	int l=prob->l;
	int w_size=get_nr_variable();
	selectiontree *T;
	Xv(w,z);
	for (i=0;i<nr_subset;i++)
	{
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);

		T=new selectiontree(nr_class[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],pi[i][k].value);
				k++;
			}
			T->count_smaller(int_y[pi[i][j].id],&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;

		T = new selectiontree(nr_class[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(int_y[pi[i][j].id],&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;
	}
	long long nSV = 0;
	for (i=0;i<l;i++)
		nSV += l_plus[i];
	info("nSV = %ld\n",nSV);
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	return(f);
}

void selection_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *wa = new double[l];
	selectiontree *T;
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		T=new selectiontree(nr_class[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],wa[pi[i][k].id]);
				k++;
			}
			alpha_plus_minus[pi[i][j].id]=T->vector_sum_smaller(int_y[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new selectiontree(nr_class[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],wa[pi[i][k].id]);
				k--;
			}
			alpha_plus_minus[pi[i][j].id]+=T->vector_sum_larger(int_y[pi[i][j].id]);
		}
		delete T;
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}

class direct_count: public selection_rank_fun
{
	public:
		direct_count(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count): selection_rank_fun(prob, C, nr_subset, perm, start, count){};
		double fun(double *w);
		void Hv(double *s, double *Hs);
};

double direct_count::fun(double *w)
{
	int i,j,k,r;
	double f=0;
	int* count_plus;
	int* count_minus;
	double* xv_plus;
	double* xv_minus;
	int l=prob->l;
	int w_size=get_nr_variable();
	Xv(w,z);
	for (i=0;i<nr_subset;i++)
	{
		count_plus = new int[nr_class[i]];
		count_minus = new int[nr_class[i]];
		xv_plus = new double[nr_class[i]];
		xv_minus = new double[nr_class[i]];
		for (j=0;j<nr_class[i];j++)
		{
			count_plus[j] = 0;
			count_minus[j] = 0;
			xv_plus[j] = 0;
			xv_minus[j] = 0;
		}
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);

		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				count_minus[int_y[pi[i][k].id]-1]++;
				xv_minus[int_y[pi[i][k].id]-1] += pi[i][k].value;
				k++;
			}
			l_minus[pi[i][j].id] = 0;
			alpha_minus[pi[i][j].id] = 0;
			for (r=0;r<int_y[pi[i][j].id]-1;r++)
			{
				l_minus[pi[i][j].id] += count_minus[r];
				alpha_minus[pi[i][j].id] += xv_minus[r];
			}
		}
		k=count[i]-1;

		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				count_plus[int_y[pi[i][k].id]-1]++;
				xv_plus[int_y[pi[i][k].id]-1] += pi[i][k].value;
				k--;
			}
			l_plus[pi[i][j].id] = 0;
			alpha_plus[pi[i][j].id] = 0;
			for (r=nr_class[i]-1;r>int_y[pi[i][j].id]-1;r--)
			{
				l_plus[pi[i][j].id] += count_plus[r];
				alpha_plus[pi[i][j].id] += xv_plus[r];
			}
		}
		delete[] count_plus;
		delete[] xv_plus;
		delete[] count_minus;
		delete[] xv_minus;
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	return(f);
}

void direct_count::Hv(double *s, double *Hs)
{
	int i,j,k,r;
	int w_size=get_nr_variable();
	int l=prob->l;
	double* xv_plus;
	double* xv_minus;
	double *wa = new double[l];
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		xv_plus = new double[nr_class[i]];
		xv_minus = new double[nr_class[i]];
		for (j=0;j<nr_class[i];j++)
		{
			xv_plus[j] = 0;
			xv_minus[j] = 0;
		}
		k=0;
		for (j=0;j<count[i];j++)
		{
			alpha_plus_minus[pi[i][j].id] = 0;
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				xv_minus[int_y[pi[i][k].id]-1] += wa[pi[i][k].id];
				k++;
			}
			for (r=0;r<int_y[pi[i][j].id]-1;r++)
				alpha_plus_minus[pi[i][j].id] += xv_minus[r];
		}
		k=count[i]-1;
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				xv_plus[int_y[pi[i][k].id]-1] +=  wa[pi[i][k].id];
				k--;
			}
			for (r=nr_class[i]-1;r>int_y[pi[i][j].id]-1;r--)
				alpha_plus_minus[pi[i][j].id] += xv_plus[r];
		}
		delete[] xv_plus;
		delete[] xv_minus;
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}

class prsvmp_fun: public y_rbtree_rank_fun
{
	public:
		prsvmp_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count);
		~prsvmp_fun();

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *s, double *Hs);
	private:
		int *int_y;
		int *nr_class;
		int **subset_perm;
		int *index_counter;
		int *class_counter;
};
prsvmp_fun::prsvmp_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count):y_rbtree_rank_fun(prob, C, nr_subset, perm, start, count)
{
	int i,j,k;
	int l=prob->l;
	this->prob = prob;
	this->nr_subset = nr_subset;
	this->perm = perm;
	this->start = start;
	this->count = count;
	this->C = C;
	double *y=prob->y;
	int_y = new int[l];
	nr_class = new int[nr_subset];
	class_counter = new int[nr_subset];
	index_counter = new int[nr_subset];
	subset_perm = new int*[nr_subset];
	for (int i=0;i<nr_subset;i++)
		subset_perm[i] = new int[count[i]];
	for (i=0;i<nr_subset;i++)
	{
		k=1;
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id=perm[j+start[i]];
			pi[i][j].value=y[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);

		int_y[pi[i][count[i]-1].id]=1;
		subset_perm[i][0] = pi[i][count[i]-1].id;
		for(j=count[i]-2;j>=0;j--)
		{
			if (pi[i][j].value>pi[i][j+1].value)
				k++;
			int_y[pi[i][j].id]=k;
			subset_perm[i][count[i]-j-1] = pi[i][j].id;
		}
		nr_class[i]=k;
	}
}

prsvmp_fun::~prsvmp_fun()
{
	delete[] int_y;
	delete[] nr_class;
	delete[] index_counter;
	delete[] class_counter;
	for (int i=0;i<nr_subset;i++)
		delete[] subset_perm[i];
	delete[] subset_perm;
}

double prsvmp_fun::fun(double *w)
{
	int i,j,k,m;
	double f=0;
	int l=prob->l;
	int w_size=get_nr_variable();
	int n;
	int n_temp;
	double alpha_sum;
	double alpha_sum_temp;
	Xv(w,z);
	for (i=0;i<l;i++)
	{
		l_plus[i] = 0;
		alpha_plus[i] = 0;
		l_minus[i] = 0;
		alpha_minus[i] = 0;
	}
	for (i=0;i<nr_subset;i++)
	{
		if (nr_class[i] > 1)
		{
			j = 0;
			while(int_y[subset_perm[i][j]] == 1)
			{
				pi[i][j].id = subset_perm[i][j];
				pi[i][j].value = z[subset_perm[i][j]]+0.5;
				j++;
			}
			for (k=2;k<=nr_class[i];k++)
			{
				while(int_y[subset_perm[i][j]] < k)
				{
					if (int_y[pi[i][j].id] == k-1)
						pi[i][j].value += 1;
					j++;
				}
				while(j<count[i] && int_y[subset_perm[i][j]] == k)
				{
					pi[i][j].id = subset_perm[i][j];
					pi[i][j].value = z[subset_perm[i][j]]-0.5;
					j++;
				}
				qsort(pi[i], j, sizeof(id_and_value), compare_id_and_value);
				n = 0;
				n_temp = 0;
				alpha_sum = 0;
				alpha_sum_temp = 0;
				if (int_y[pi[i][0].id] < k)
				{
					n_temp++;
					alpha_sum_temp += pi[i][0].value;
				}
				else
				{
					alpha_minus[pi[i][0].id] += alpha_sum;
					l_minus[pi[i][0].id] += n;
				}
				for (m=1;m<j;m++)
				{
					if (pi[i][m].value < pi[i][m-1].value)
					{
						n += n_temp;
						n_temp = 0;
						alpha_sum += alpha_sum_temp;
						alpha_sum_temp = 0;
					}
					if (int_y[pi[i][m].id] < k)
					{
						n_temp++;
						alpha_sum_temp += pi[i][m].value;
					}
					else
					{
						alpha_minus[pi[i][m].id] += alpha_sum;
						l_minus[pi[i][m].id] += n;
					}
				}
				n = 0;
				n_temp = 0;
				alpha_sum = 0;
				alpha_sum_temp = 0;
				m = j - 1;
				if (int_y[pi[i][m].id] == k)
				{
					n_temp++;
					alpha_sum_temp += pi[i][m].value;
				}
				else
				{
					alpha_plus[pi[i][m].id] += alpha_sum;
					l_plus[pi[i][m].id] += n;
				}
				m--;
				for (;m>=0;m--)
				{
					if (pi[i][m].value > pi[i][m+1].value)
					{
						n += n_temp;
						n_temp = 0;
						alpha_sum += alpha_sum_temp;
						alpha_sum_temp = 0;
					}
					if (int_y[pi[i][m].id] == k)
					{
						n_temp++;
						alpha_sum_temp += pi[i][m].value;
					}
					else
					{
						alpha_plus[pi[i][m].id] += alpha_sum;
						l_plus[pi[i][m].id] += n;
					}
				}
				j = 0;
			}
			index_counter[i] = count[i];
			class_counter[i] = nr_class[i] - 2;
		}
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
		f += C*((z[i]+0.5)*((z[i]+0.5)*l_plus[i]-alpha_plus[i])+(z[i]-0.5)*((z[i]-0.5)*l_minus[i]-alpha_minus[i]));
	return(f);
}

void prsvmp_fun::grad(double *w, double *g)
{
	int l = prob->l;
	int i;
	int w_size=get_nr_variable();
	double *dLdy = new double[l];
	for (i=0;i<l;i++)
		dLdy[i] = l_plus[i]*(z[i]+0.5)+l_minus[i]*(z[i]-0.5)-(alpha_plus[i]+alpha_minus[i]);
	XTv(dLdy,g);
	for (i=0;i<w_size;i++)
		g[i] = w[i] + 2 * C * g[i];
}

void prsvmp_fun::Hv(double *s, double *Hs)
{
	int i,j,k,m,n;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *wa = new double[l];
	double* gamma_plus_minus;
	double gamma_sum;
	double gamma_temp;
	gamma_plus_minus = new double[l];
	for (i=0;i<l;i++)
		gamma_plus_minus[i] = 0;
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		if (nr_class[i] > 1)
		{
			for (n=0;n<nr_class[i]-1;n++)
			{
				gamma_temp = 0;
				gamma_sum = 0;
				k = 2 +(n + class_counter[i]) % (nr_class[i] - 1);
				if (n == 0)
					j = index_counter[i];
				else
				{
					j = 0;
					if (k == 2)
					{
						while(int_y[subset_perm[i][j]] == 1)
						{
							pi[i][j].id = subset_perm[i][j];
							pi[i][j].value = z[subset_perm[i][j]]+0.5;
							j++;
						}
					}
					else
					{
						while(int_y[subset_perm[i][j]] < k)
						{
							if(int_y[pi[i][j].id] == k-1)
								pi[i][j].value += 1.0;
							j++;
						}
					}
					while(j<count[i] && int_y[subset_perm[i][j]] == k)
					{
						pi[i][j].id = subset_perm[i][j];
						pi[i][j].value = z[subset_perm[i][j]]-0.5;
						j++;
					}
					qsort(pi[i], j, sizeof(id_and_value), compare_id_and_value);
				}

				if (int_y[pi[i][0].id] < k)
					gamma_temp += wa[pi[i][0].id];
				for (m=1;m<j;m++)
				{
					if (pi[i][m].value < pi[i][m-1].value)
					{
						gamma_sum += gamma_temp;
						gamma_temp = 0;
					}
					if (int_y[pi[i][m].id] < k)
						gamma_temp += wa[pi[i][m].id];
					else
						gamma_plus_minus[pi[i][m].id] += gamma_sum;
				}
				gamma_sum = 0;
				gamma_temp = 0;
				m = j - 1;
				if (int_y[pi[i][m].id] == k)
					gamma_temp += wa[pi[i][m].id];
				m--;
				for (;m>=0;m--)
				{
					if (pi[i][m].value > pi[i][m+1].value)
					{
						gamma_sum += gamma_temp;
						gamma_temp = 0;
					}
					if (int_y[pi[i][m].id] == k)
						gamma_temp += wa[pi[i][m].id];
					else
						gamma_plus_minus[pi[i][m].id] += gamma_sum;
				}
			}
			index_counter[i] = j;
			class_counter[i] = k-2;
		}
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-gamma_plus_minus[i];
	delete[] gamma_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}

// To support weights for instances, use GETI(i) (i)

// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = C
// 		lambda_i = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr(
		const problem *prob, double *w, const parameter *param,
		int solver_type)
{
	int l = prob->l;
	double C = param->C;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init;
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = INF;

	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = 0;
		feature_node *xi = prob->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			w[xi->index-1] += beta[i]*val;
			xi++;
		}

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			feature_node *xi = prob->x[i];
			while(xi->index != -1)
			{
				int ind = xi->index-1;
				double val = xi->value;
				G += val*w[ind];
				xi++;
			}

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
			{
				xi = prob->x[i];
				while(xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: MAX ITERATION REACHED\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);

	delete [] beta;
	delete [] QD;
	delete [] index;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void group_queries(const problem *prob, int *nr_subset_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_subset = 16;
	int nr_subset = 0;
	int *query = Malloc(int,max_nr_subset);
	int *count = Malloc(int,max_nr_subset);
	int *data_query = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_query = (int)prob->query[i];
		int j;
		for(j=0;j<nr_subset;j++)
		{
			if(this_query == query[j])
			{
				++count[j];
				break;
			}
		}
		data_query[i] = j;
		if(j == nr_subset)
		{
			if(nr_subset == max_nr_subset)
			{
				max_nr_subset *= 2;
				query = (int *)realloc(query,max_nr_subset*sizeof(int));
				count = (int *)realloc(count,max_nr_subset*sizeof(int));
			}
			query[nr_subset] = this_query;
			count[nr_subset] = 1;
			++nr_subset;
		}
	}

	int *start = Malloc(int,nr_subset);
	start[0] = 0;
	for(i=1;i<nr_subset;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_query[i]]] = i;
		++start[data_query[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_subset;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_subset_ret = nr_subset;
	*start_ret = start;
	*count_ret = count;
	free(query);
	free(data_query);
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn, int nr_subset=0, int *perm=NULL, int *start=NULL, int *count=NULL)
{
	double eps=param->eps;
	int pos = 0;
	int neg = 0;
	clock_t begin,end;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = prob->l - pos;

	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;

	function *fun_obj=NULL;
	begin = clock();
	switch(param->solver_type)
	{
		case L2R_L2LOSS_SVC:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete C;
			break;
		}
		case L2R_L2LOSS_SVR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
				C[i] = param->C;
			fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
			TRON tron_obj(fun_obj, param->eps);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete C;
			break;
		}
		case Y_RBTREE:
			{
				fun_obj=new y_rbtree_rank_fun(prob, param->C, nr_subset, perm, start, count);
				TRON tron_obj(fun_obj, param->eps);
				tron_obj.set_print_string(liblinear_print_string);
				tron_obj.tron(w);
				delete fun_obj;
				break;
			}
		case WX_RBTREE:
			{
				fun_obj=new wx_rbtree_rank_fun(prob, param->C, nr_subset, perm, start, count);
				TRON tron_obj(fun_obj, param->eps);
				tron_obj.set_print_string(liblinear_print_string);
				tron_obj.tron(w);
				delete fun_obj;
				break;
			}
		case SELECTION_TREE:
			{
				fun_obj=new selection_rank_fun(prob, param->C, nr_subset, perm, start, count);
				TRON tron_obj(fun_obj, param->eps);
				tron_obj.set_print_string(liblinear_print_string);
				tron_obj.tron(w);
				delete fun_obj;
				break;
			}
		case DIRECT_COUNT:
			{
				fun_obj=new direct_count(prob, param->C, nr_subset, perm, start, count);
				TRON tron_obj(fun_obj, param->eps);
				tron_obj.set_print_string(liblinear_print_string);
				tron_obj.tron(w);
				delete fun_obj;
				break;
			}
		case AVLTREE:
			{
				fun_obj=new y_avltree_rank_fun(prob, param->C, nr_subset, perm, start, count);
				TRON tron_obj(fun_obj, param->eps);
				tron_obj.set_print_string(liblinear_print_string);
				tron_obj.tron(w);
				delete fun_obj;
				break;
			}
		case AATREE:
			{
				fun_obj=new y_aatree_rank_fun(prob, param->C, nr_subset, perm, start, count);
				TRON tron_obj(fun_obj, param->eps);
				tron_obj.set_print_string(liblinear_print_string);
				tron_obj.tron(w);
				delete fun_obj;
				break;
			}
		case PRSVMP:
			{
				fun_obj=new prsvmp_fun(prob, param->C, nr_subset, perm, start, count);
				TRON tron_obj(fun_obj, param->eps);
				tron_obj.set_print_string(liblinear_print_string);
				tron_obj.tron(w);
				delete fun_obj;
				break;
			}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL);
			break;
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}
	end = clock();
	info("Training time = %g\n",double(end-begin)/double(CLOCKS_PER_SEC));
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	model_->nr_feature=n;
	model_->param = *param;

	if(param->solver_type == L2R_L2LOSS_SVR ||
			param->solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		model_->w = Malloc(double, w_size);
		model_->nr_class = 2;
		model_->label = NULL;
		train_one(prob, param, &model_->w[0], 0, 0);
	}
	else if(param->solver_type == WX_RBTREE||
			param->solver_type == Y_RBTREE||
			param->solver_type == SELECTION_TREE||
			param->solver_type == AVLTREE||
			param->solver_type == AATREE||
			param->solver_type == DIRECT_COUNT||
			param->solver_type == PRSVMP)
	{
		model_->w = Malloc(double, w_size);
		model_->nr_class = 2;
		model_->label = NULL;
		int nr_subset;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);
		group_queries(prob, &nr_subset ,&start, &count, perm);
		train_one(prob, param, &model_->w[0],0,0, nr_subset, perm, start, count);
		free(start);
		free(count);
		free(perm);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		if(nr_class == 2)
		{
			model_->w=Malloc(double, w_size);

			int e0 = start[0]+count[0];
			k=0;
			for(; k<e0; k++)
				sub_prob.y[k] = +1;
			for(; k<sub_prob.l; k++)
				sub_prob.y[k] = -1;

			train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
		}
		else
		{
			model_->w=Malloc(double, w_size*nr_class);
			double *w=Malloc(double, w_size);
			for(i=0;i<nr_class;i++)
			{
				int si = start[i];
				int ei = si+count[i];

				k=0;
				for(; k<si; k++)
					sub_prob.y[k] = -1;
				for(; k<ei; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				train_one(&sub_prob, param, w, weighted_C[i], param->C);

				for(int j=0;j<w_size;j++)
					model_->w[j*nr_class+i] = w[j];
			}
			free(w);
		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
		free(weighted_C);
	}
	return model_;
}

static void group_queries(const int *query_id, int l, int *nr_query_ret, int **start_ret, int **count_ret, int *perm)
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

void eval_list(double *label, double *target, int *query, int l, double *result_ret)
{
	int q,i,j,k;
	int nr_query;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int, l);
	id_and_value *order_perm;
	int true_query;
	int ndcg_size;
	long long totalnc = 0, totalnd = 0;
	long long nc = 0;
	long long nd = 0;
	double tmp;
	double accuracy = 0;
	int *l_plus;
	int *int_y;
	int same_y = 0;
	double *ideal_dcg;
	double *dcg;
	double meanndcg = 0;
	double ndcg;
	double dcg_yahoo,idcg_yahoo,ndcg_yahoo;
	selectiontree *T;
	group_queries(query, l, &nr_query, &start, &count, perm);
	true_query = nr_query;
	for (q=0;q<nr_query;q++)
	{
		//We use selection trees to compute pairwise accuracy
		nc = 0;
		nd = 0;
		l_plus = new int[count[q]];
		int_y = new int[count[q]];
		order_perm = new id_and_value[count[q]];
		int *perm_q = &perm[start[q]];
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = i;
			order_perm[i].value = label[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		int_y[order_perm[count[q]-1].id] = 1;
		same_y = 0;
		k = 1;
		for(i=count[q]-2;i>=0;i--)
		{
			if (order_perm[i].value != order_perm[i+1].value)
			{
				same_y = 0;
				k++;
			}
			else
				same_y++;
			int_y[order_perm[i].id] = k;
			nc += (count[q]-1 - i - same_y);
		}
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = i;
			order_perm[i].value = target[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		//total pairs
		T = new selectiontree(k);
		j = count[q] - 1;
		for (i=count[q] - 1;i>=0;i--)
		{
			while (j>=0 && ( order_perm[j].value < order_perm[i].value))
			{
				T->insert_node(int_y[order_perm[j].id], tmp);
				j--;
			}
			T->count_larger(int_y[order_perm[i].id], &l_plus[order_perm[i].id], &tmp);
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
		delete[] order_perm;
	}
	result_ret[0] = (double)totalnc/(double)(totalnc+totalnd);
	for (q=0;q<nr_query;q++)
	{
		ndcg_size = min(10,count[q]);
		ideal_dcg = new double[count[q]];
		dcg = new double[count[q]];
		ndcg = 0;
		order_perm = new id_and_value[count[q]];
		int *perm_q = &perm[start[q]];
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = perm_q[i];
			order_perm[i].value = label[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		ideal_dcg[0] = pow(2.0,order_perm[0].value) - 1;
		idcg_yahoo = pow(2.0, order_perm[0].value) - 1;
		for (i=1;i<count[q];i++)
			ideal_dcg[i] = ideal_dcg[i-1] + (pow(2.0,order_perm[i].value) - 1) * log(2.0) / log(i+1.0);
		for (i=1;i<ndcg_size;i++)
			idcg_yahoo += (pow(2.0,order_perm[i].value) - 1) * log(2.0) / log(i+2.0);
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = perm_q[i];
			order_perm[i].value = target[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		dcg[0] = pow(2.0, label[order_perm[0].id]) - 1;
		dcg_yahoo = pow(2.0, label[order_perm[0].id]) - 1;
		for (i=1;i<count[q];i++)
			dcg[i] = dcg[i-1] + (pow(2.0, label[order_perm[i].id]) - 1) * log(2.0) / log(i + 1.0);
		for (i=1;i<ndcg_size;i++)
			dcg_yahoo += (pow(2.0, label[order_perm[i].id]) - 1) * log(2.0) / log(i + 2.0);
		if (ideal_dcg[0]>0)
			for (i=0;i<count[q];i++)
				ndcg += dcg[i]/ideal_dcg[i];
		else
			ndcg = 0;
		meanndcg += ndcg/count[q];
		delete[] order_perm;
		delete[] ideal_dcg;
		delete[] dcg;
		if (idcg_yahoo > 0)
			ndcg_yahoo += dcg_yahoo/idcg_yahoo;
		else
			ndcg_yahoo += 1;
	}
	meanndcg /= nr_query;
	ndcg_yahoo /= nr_query;
	result_ret[1] = meanndcg;
	result_ret[2] = ndcg_yahoo;
	free(start);
	free(count);
	free(perm);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}
	return dec_values[0];
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

static const char *solver_type_table[]=
{
"L2R_L2LOSS_SVC", "L2R_L2LOSS_SVR", "L2R_L1LOSS_SVR_DUAL", "DIRECT_COUNT","Y_RBTREE","WX_RBTREE","SELECTION_TREE","AVLTREE","AATREE","PRSVMP",NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");

				setlocale(LC_ALL, old_locale);
				free(model_->label);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			setlocale(LC_ALL, old_locale);
			free(model_->label);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_L2LOSS_SVC
			&& param->solver_type != L2R_L2LOSS_SVR
			&& param->solver_type != L2R_L1LOSS_SVR_DUAL
			&& param->solver_type != Y_RBTREE
			&& param->solver_type != WX_RBTREE
			&& param->solver_type != SELECTION_TREE
			&& param->solver_type != AVLTREE
			&& param->solver_type != AATREE
			&& param->solver_type != DIRECT_COUNT
			&& param->solver_type != PRSVMP)
		return "unknown solver type";

	return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

