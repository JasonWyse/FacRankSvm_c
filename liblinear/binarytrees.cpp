#include "binarytrees.h"
#include <cstddef>
#include <stdio.h>
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
rbtree::rbtree(int l)
{
	null_node = new node;
	null_node->size = 0;
	null_node->vector_sum = 0;
	null_node->color = BLACK;
	this->tree_size = 0;
	this->tree_nodes = new node[l];
	this->root=null_node;
}

rbtree::~rbtree()
{
	delete null_node;
	delete[] tree_nodes;
}

void rbtree::insert_node(double key, double value)
{
	node* new_node;
	new_node = &this->tree_nodes[this->tree_size];
	this->tree_size++;
	if (this->root == null_node)
	{
		this->root = new_node;
		new_node->parent = null_node;
		new_node->color = BLACK;
	}
	else
	{
		node* x = root;
		while(1)
		{
			x->size++;
			x->vector_sum += value;
			if (key == x->key)
			{
				this->tree_size--;
				return;
			}
			if (key > x->key)
			{
				if (x->child[RIGHT] != null_node)
					x = x->child[RIGHT];
				else
				{
					x->child[RIGHT] = new_node;
					break;
				}
			}
			else
			{
				if (x->child[LEFT] != null_node)
					x = x->child[LEFT];
				else
				{
					x->child[LEFT] = new_node;
					break;
				}
			}
		}
		new_node->parent = x;
		new_node->color = RED;
	}
	new_node->key = key;
	new_node->vector_sum = value;
	new_node->size = 1;
	new_node->child[LEFT] = null_node;
	new_node->child[RIGHT] = null_node;
	tree_color_fix(new_node);
}

void rbtree::count_smaller(double key,int* count_ret, double* acc_value_ret)
{
	int count = 0;
	double acc_value = 0;
	if (this->tree_size == 0)
	{
		*count_ret = 0;
		*acc_value_ret = 0;
		return;
	}
	node* x = root;
	while (x != null_node)
	{
		if (key == x->key)
		{
			count += x->child[LEFT]->size;
			acc_value += x->child[LEFT]->vector_sum;
			break;
		}
		if (key > x->key)
		{
			count += x->size - x->child[RIGHT]->size;
			acc_value += x->vector_sum - x->child[RIGHT]->vector_sum;
			x = x->child[RIGHT];
		}
		else
			x = x->child[LEFT];
	}
	*count_ret = count;
	*acc_value_ret = acc_value;
	return;
}

void rbtree::count_larger(double key, int* count_ret, double* acc_value_ret)
{
	int count = 0;
	double acc_value = 0;
	if (this->tree_size == 0)
	{
		*count_ret = 0;
		*acc_value_ret = 0;
		return;
	}
	node* x = root;
	while (x != null_node)
	{
		if (key == x->key)
		{
			count += x->child[RIGHT]->size;
			acc_value += x->child[RIGHT]->vector_sum;
			break;
		}
		if (key < x->key)
		{
			count += x->size - x->child[LEFT]->size;
			acc_value += x->vector_sum - x->child[LEFT]->vector_sum;
			x = x->child[LEFT];
		}
		else
			x = x->child[RIGHT];
	}
	*count_ret = count;
	*acc_value_ret = acc_value;
	return;
}

double rbtree::vector_sum_larger(double key)
{
	double acc_value = 0;
	if (this->tree_size == 0)
		return 0;
	node* x = root;
	while (x != null_node)
	{
		if (key == x->key)
		{
			acc_value += x->child[RIGHT]->vector_sum;
			return acc_value;
		}
		if (key < x->key)
		{
			acc_value += x->vector_sum - x->child[LEFT]->vector_sum;
			x = x->child[LEFT];
		}
		else
			x = x->child[RIGHT];
	}
	return acc_value;

}
double rbtree::vector_sum_smaller(double key)
{
	double acc_value = 0;
	if (this->tree_size == 0)
		return 0;
	node* x = root;
	while (x != null_node)
	{
		if (key == x->key)
		{
			acc_value += x->child[LEFT]->vector_sum;
			return acc_value;
		}
		if (key > x->key)
		{
			acc_value += x->vector_sum - x->child[RIGHT]->vector_sum;
			x = x->child[RIGHT];
		}
		else
			x = x->child[LEFT];
	}
	return acc_value;

}
void rbtree::rotate(node* x, int direction)
{
	node* y = x->child[!direction];
	if (x != root)
	{
		int self_direction;
		if (x->parent->child[LEFT] == x)
			self_direction=LEFT;
		else
			self_direction=RIGHT;

		x->parent->child[self_direction] = y;
		y->parent = x->parent;
	}
	else
	{
		y->parent = null_node;
		root = y;
	}
	x->size -= y->size;
	x->vector_sum -= y->vector_sum;
	y->size += x->size;
	y->vector_sum += x->vector_sum;
	x->parent = y;
	x->child[!direction] = y->child[direction];
	y->child[direction] = x;
	if (x->child[!direction] != null_node)
	{
		x->child[!direction]->parent = x;
		x->size += x->child[!direction]->size;
		x->vector_sum += x->child[!direction]->vector_sum;
	}
}

void rbtree::tree_color_fix(node* x)
{
	node* y=x;
	int uncle_direction, self_direction;
	while(y->parent != null_node)
	{
		if (y->parent->color == BLACK)
			return;
		else
		{
			node* z = y->parent;
			if (z->key < z->parent->key) //z is left child
				uncle_direction = RIGHT;
			else
				uncle_direction = LEFT;
			z = z->parent->child[uncle_direction];//now z is uncle of y

			if (z->color == RED)
			{
				z->color = BLACK;
				y = y->parent;
				y->color = BLACK;
				y = y->parent;
				y->color = RED;
			}
			else
			{
				if (y->key > y->parent->key)// y is right child
					self_direction=RIGHT;
				else
					self_direction=LEFT;
				if (self_direction == uncle_direction)
					rotate(y->parent, !self_direction);
				else
					y = y->parent;
				y->color = BLACK;
				y->parent->color = RED;
				rotate(y->parent, uncle_direction);
				return;
			}
		}
	}
	y->color = BLACK;
	this->root = y;
	y->parent = null_node;
}

avl::avl(int l) : rbtree(l)
{
	null_node->height = 0;
}

void avl::insert_node(double key, double value)
{
	node* new_node;
	new_node = &this->tree_nodes[this->tree_size];
	this->tree_size++;
	if (this->root == null_node)
	{
		this->root = new_node;
		new_node->parent = null_node;
		new_node->height = 0;
	}
	else
	{
		node* x = root;
		while(1)
		{
			x->size++;
			x->vector_sum += value;
			if (key == x->key)
			{
				this->tree_size--;
				return;
			}
			if (key > x->key)
			{
				if (x->child[RIGHT] != null_node)
					x = x->child[RIGHT];
				else
				{
					x->child[RIGHT] = new_node;
					break;
				}
			}
			else
			{
				if (x->child[LEFT] != null_node)
					x = x->child[LEFT];
				else
				{
					x->child[LEFT] = new_node;
					break;
				}
			}
		}
		new_node->parent = x;
		new_node->height = 0;
	}
	new_node->key = key;
	new_node->vector_sum = value;
	new_node->size = 1;
	new_node->child[LEFT] = null_node;
	new_node->child[RIGHT] = null_node;
	tree_balance_fix(new_node);
}

void avl::tree_balance_fix(node* x)
{
	int balance_factor;
	int sub_balance;
	node* y=x;
	while(y->parent != null_node)
	{
		y->height = max(y->child[LEFT]->height,y->child[RIGHT]->height) + 1;
		if (y->height != 0)
			balance_factor = y->child[LEFT]->height - y->child[RIGHT]->height;
		else
			balance_factor = 0;

		if (balance_factor > 1)
		{
			node* z = y->child[LEFT];
			sub_balance =  z->child[LEFT]->height - z->child[RIGHT]->height;
			if (sub_balance == -1)
			{
				rotate(z,LEFT);
				z->height = max(z->child[LEFT]->height, z->child[RIGHT]->height) + 1;
				z->parent->height = max(z->height,z->parent->child[RIGHT]->height)+1;
			}
			if (sub_balance != 0)
			{
				rotate(y,RIGHT);
				y->height = max(y->child[LEFT]->height, y->child[RIGHT]->height) + 1;
				y->parent->height = max(y->height,y->parent->child[LEFT]->height)+1;
				y = y->parent;
			}
		}
		else if(balance_factor < -1)
		{
			node* z = y->child[RIGHT];
			sub_balance =  z->child[LEFT]->height - z->child[RIGHT]->height;
			if (sub_balance == 1)
			{
				rotate(z,RIGHT);
				z->height = max(z->child[LEFT]->height, z->child[RIGHT]->height) + 1;
				z->parent->height = max(z->height,z->parent->child[LEFT]->height)+1;
			}
			if (sub_balance != 0)
			{
				rotate(y,LEFT);
				y->height = max(y->child[LEFT]->height, y->child[RIGHT]->height) + 1;
				y->parent->height = max(y->height,y->parent->child[RIGHT]->height)+1;
				y = y->parent;
			}
		}
		y = y->parent;
	}
}

void aatree::tree_color_fix(node* x)
{
	node* y = x->parent;
	node* z = y;
	while(z != null_node)
	{
		y = z;
		if (y->child[LEFT]->color == RED) // skew
		{
			y->color = RED;
			y->child[LEFT]->color = BLACK;
			rotate(y,RIGHT);
		}
		else if (y->color == RED && y->child[RIGHT]->color == RED) // split
		{
			y->child[RIGHT]->color = BLACK;
			rotate(y->parent,LEFT);
			z = y->parent;
		}
		else
			return;
	}
	y->color = BLACK;
	this->root = y;
	y->parent = null_node;
}
