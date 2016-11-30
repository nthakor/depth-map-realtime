#include <stdio.h>
#include <iostream>
#include <string.h>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
// #include <vector>
using namespace cv;
using namespace std;

// typedef typename std::vector<T>::iterator iterator;

#define MAX_DIM 32
#define threshold 10

struct kd_node_t{
    double x[MAX_DIM];
    struct kd_node_t *left, *right;
};

///////////////////**************function definitions**************////////////////////////////////////////
double dist(struct kd_node_t *a, struct kd_node_t *b, int dim)
{
    double t, d = 0;
    while (dim--) {
        // cout<<"dim:"<<dim<<"\n"<<endl;
        t = a->x[dim] - b->x[dim];
        d += t * t;
    }
    return d;
}
void swap(struct kd_node_t *x, struct kd_node_t *y) {
    double tmp[MAX_DIM];
    memcpy(tmp,  x->x, sizeof(tmp));
    memcpy(x->x, y->x, sizeof(tmp));
    memcpy(y->x, tmp,  sizeof(tmp));
}


/* see quickselect method */
struct kd_node_t* find_median(struct kd_node_t *start, struct kd_node_t *end, int idx)
{
    if (end <= start) return NULL;
    if (end == start + 1)
        return start;
    
    struct kd_node_t *p, *store, *md = start + (end - start) / 2;
    double pivot;
    while (1) {
        pivot = md->x[idx];
        
        swap(md, end - 1);
        for (store = p = start; p < end; p++) {
            if (p->x[idx] < pivot) {
                if (p != store)
                    swap(p, store);
                store++;
            }
        }
        swap(store, end - 1);
        
        /* median has duplicate values */
        if (store->x[idx] == md->x[idx])
            return md;
        
        if (store > md) end = store;
        else        start = store;
    }
}

struct kd_node_t* make_tree(struct kd_node_t *t, int len, int i, int dim)
{
    struct kd_node_t *n;
    
    if (!len) return NULL;
    
    if ((n = find_median(t, t + len, i))) {
        i = (i + 1) % dim;

        n->left  = make_tree(t, n - t, i, dim);
        n->right = make_tree(n + 1, t + len - (n + 1), i, dim);
    }
    return n;
}

/* global variable, so sue me */
int visited;

void nearest(struct kd_node_t *root, struct kd_node_t *nd, int i, int dim,
             struct kd_node_t **best, double *best_dist)
{
    double d, dx, dx2;
    
    if (root==NULL) return;
    d = dist(root, nd, dim);
    dx = root->x[i] - nd->x[i];
    dx2 = dx * dx;
    
    visited ++;
    
    if (!*best || d < *best_dist) {
        *best_dist = d;
        *best = root;
    }
    
    /* if chance of exact match is high */
    if (!*best_dist) return;
    
    if (++i >= dim) i = 0;
    
    nearest(dx > 0 ? root->left : root->right, nd, i, dim, best, best_dist);
    if (dx2 >= *best_dist) return;
    nearest(dx > 0 ? root->right : root->left, nd, i, dim, best, best_dist);
}

int Nodei=0;

void postorder(struct kd_node_t *p, int indent=0)
{
    if(p != NULL) {
        if(p->left) postorder(p->left, indent+4);
        if(p->right) postorder(p->right, indent+4);
        if (indent) {
            std::cout << std::setw(indent) << ' ';
        }
        cout<< Nodei << "\n ";
        Nodei++;
    }
}


void walkTree(struct kd_node_t *root){
    // cout<<"Entering"<<endl;
    if(root==NULL){
        // cout<<"NULL";
        return;
    }
    else
    {
        // cout<<"No IF"<<endl;
        // cout<<"left"<<endl;
        // cout<<"print"<<endl;
        walkTree(root->left);
        for (int i = 0; i < MAX_DIM; ++i)
        {
            cout<<root->x[i]<<"\t";   
        }
        cout<<"Nodei"<<Nodei<<"\n"<<endl;
        Nodei++;
        // cout<<"\nright"<<endl;
        walkTree(root->right);
    }

}
///////////////////////////////////////////////////////////////////

    Ptr<DescriptorExtractor> descriptorExtractor = ORB::create();
    Ptr<FeatureDetector> detector = ORB::create();

    // Mat descriptors;
    // Mat Desc;
int main()
{   
    struct kd_node_t *root, *found, *million;
    struct kd_node_t testNode;
    int i; double best_dist;   
    Mat frame;
    Mat gray;
    int counter=0;
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
       return -1;
    namedWindow("gray",1);
    for(;;)
    {
        Mat desc;
        Mat descriptors;
        cap >> frame; // get a new frame from camera
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<KeyPoint> kp;
        vector<int> kp_show;
        detector->detect(gray, kp);
        descriptorExtractor->compute(gray, kp, descriptors);
        // cout<<descriptors.rows<<"\t"<<kp.size()<<endl;
        descriptors.convertTo(desc, CV_32FC1);
        if(kp.size()<200 && counter==0){
             cout<<"Not Entering\n"<<endl;
        }
        else if (counter==0)
        {
            struct kd_node_t wp[descriptors.rows];
            for (int j = 0; j< descriptors.rows; ++j)
            {
                for (i = 0; i < descriptors.cols; ++i)
                {
                    wp[j].x[i] = desc.at<float>(j,i);
                }
            
            }
            // cout<<"wpLen : %zu rows: %d\n",sizeof(wp) / sizeof(wp[1]),descriptors.rows);
            root = make_tree(wp, sizeof(wp) / sizeof(wp[1]), 0, descriptors.cols);
            postorder(root);
            cout<<"Total Rows:"<<descriptors.rows<<"\n"<<endl;
            // cout<<"Root done\n"<<endl; 
            counter++;
            // for(int j=0; j<descriptors.rows; j++){
            // for (i = 0; i < 32; ++i)
            //     {   
            //         //cout<<"des:"<<Desc.at<float>(j,i)<<"\n"<<endl;
            //         testNode.x[i] = desc.at<float>(j,i);
            //     }
            // }
            // nearest(root, &testNode, 0, MAX_DIM, &found, &best_dist);
            // cout<<"Done"<<endl;
        }
        else
        {   
            int count = 0;
            cout<<"frame"<<counter<<"\n"<<endl;
            cout<<"Total Rows:"<<descriptors.rows<<"\n"<<endl;
            for (int j = 0; j< descriptors.rows; ++j)
            {
                for (i = 0; i < 32; ++i)
                {   
                    //cout<<"des:"<<Desc.at<float>(j,i)<<"\n"<<endl;
                    testNode.x[i] = desc.at<float>(j,i);
                }
                visited=0;
                found=0;
                cout<<"walkDone";
                Nodei=0;
                walkTree(root);
                nearest(root, &testNode, 0, MAX_DIM, &found, &best_dist);
                //cout<<"Node"<<j<<"best_dist:"<<best_dist<<"\n"<<endl;
                // if (best_dist < threshold)
                // {   
                    // cout<<kp.at(j)<<"\n"<<endl;
                    // vector<T>::const_iterator first = kp.begin() + j;
                    // vector<T>::const_iterator last = kp.begin() + j+1;
                    // vector<T> newVec(first, last);
                    // drawKeypoints(gray,newVec,gray);
                    // kp_show.push_back(j);
                    // count++; ////problem
                // }
            }
            //descriptors.convertTo(descriptors, CV_8U);
            // root = make_tree(wp, sizeof(wp) / sizeof(wp[1]), 0, descriptors.cols);
            counter++;
            descriptors.release();
            desc.release();
        }
        drawKeypoints(gray, kp, gray);
        imshow("gray", gray);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}
        