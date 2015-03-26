enum {RED,BLACK};
enum {LEFT,RIGHT};
struct node
{
	node* parent;
	node* child[2];
	double key;
	int size;
	bool color;
	int height;
	double vector_sum;
};

class rbtree
{
public:
	rbtree(int l);
	~rbtree();
	void insert_node(double key, double value);
	void count_larger(double key, int* count_ret, double* acc_value_ret);
	void count_smaller(double key, int* count_ret, double* acc_value_ret);
	double vector_sum_larger(double key);
	double vector_sum_smaller(double key);
	int get_size(){ return tree_size;}
protected:
	node* null_node;
	int tree_size;
	void rotate(node* x, int direction);
	void tree_color_fix(node* x);
	node* root;
	node* tree_nodes;
};


class avl: public rbtree
{
public:
	avl(int l);
	void insert_node(double key, double value);
private:
	void tree_balance_fix(node* x);
};

class aatree: public rbtree
{
public:
	aatree(int l):rbtree(l){};
private:
	void tree_color_fix(node* x);
};

