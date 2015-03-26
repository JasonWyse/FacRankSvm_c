struct tree_node
{
	double xv;
	int size;
};

class selectiontree
{
public:
	selectiontree(int k); //k leaves
	~selectiontree();
	void insert_node(int key, double value);
	void delete_node(int key, double value);
	void larger(int key, int *l_plus_ret, double *alpha_plus_ret);
	void smaller(int key, int *l_minus_ret, double *alpha_minus_ret);
	double xv_larger(int key);
	double xv_smaller(int key);

private:
	int node_size;
	int leaf_size;
	tree_node *node;
};

