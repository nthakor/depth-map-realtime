#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "omp.h"

using namespace cv;
using namespace std;

#define MAX_DIM 32
#define threshold 1500

struct kd_node_t{
    double x[MAX_DIM];
    struct kd_node_t *left=NULL, *right=NULL;
};

int Nodei=0;
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


///////////////////**************function definitions**************////////////////////////////////////////
double dist(struct kd_node_t *a, struct kd_node_t *b, int dim)
{
    double t, d = 0;
    while (dim--) {
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
        #pragma omp single nowait
        {
            #pragma omp task
            n->left  = make_tree(t, n - t, i, dim);
            #pragma omp task
            n->right = make_tree(n + 1, t + len - (n + 1), i, dim);
        }
        #pragma omp taskwait   
    }

    return n;
}

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
    
    // #pragma omp single
    // {    
        // #pragma omp task
        nearest(dx > 0 ? root->left : root->right, nd, i, dim, best, best_dist);
    // }
    // #pragma omp taskwait

        if (dx2 >= *best_dist) return;
    
    // #pragma omp single
    // {   
        // #pragma omp task
        nearest(dx > 0 ? root->right : root->left, nd, i, dim, best, best_dist);

    // }
    // #pragma omp taskwait
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
    int i; double best_dist;int count_des;
    int *p;//*p=0;   
    Mat frame;
    Mat gray;
    int counter=0;
    string filename = "vtest.avi";
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
       return -1;
    namedWindow("gray",1);
    Mat desc;
    Mat desc_show;
    Mat desc_show2 = Mat(1000,32,CV_32FC1);
    Mat descriptors;
    Mat descriptors_show;
    vector<KeyPoint> kp;
    vector<KeyPoint> kp_show;
    float temp;
    // double start_time = omp_get_wtime();
    // #pragma omp parallel shared(counter) private(count_des,frame,kp,descriptors,desc)
    // {
    // #pragma omp parallel for schedule(dynamic) nowait
    for(;;)
    {
        cap >> frame; // get a new frame from camera
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if(counter==0)
        {
            Ptr<DescriptorExtractor> descriptorExtractor = ORB::create();
            Ptr<FeatureDetector> detector = ORB::create();
            detector->detect(gray, kp_show);
            descriptorExtractor->compute(gray, kp_show, descriptors_show);    
            descriptors_show.convertTo(desc_show, CV_32FC1);
            count_des = descriptors_show.rows;
        }
        detector->detect(gray, kp);
        descriptorExtractor->compute(gray, kp, descriptors);
        descriptors.convertTo(desc, CV_32FC1);
        
        double start_time = omp_get_wtime();

        if(kp.size()<200 && counter==0)
        {
             cout<<"Not Entering\n"<<endl;
        }
        else
        {   
                    // cout<<"show:"<<descriptors_show.rows;
                    // break;
            // counter = counter + 1;
            if(counter==1)
            {
                for (int k = 0; k < desc_show.rows; ++k)
                {
                    for(i = 0;i < desc_show.cols; i++)
                    {
                        desc_show2.at<float>(k,i) = desc_show.at<float>(k,i);
                    }
                }
            }

            struct kd_node_t wp[count_des];
            for (int j = 0; j< count_des; ++j)
            {
               for (i = 0; i < desc_show2.cols; ++i)
                    {
                        wp[j].x[i] = desc_show2.at<float>(j,i);
                    }
            }
            #pragma omp parallel 
            {
                root = make_tree(wp, sizeof(wp) / sizeof(wp[1]), 0, descriptors.cols);
                /****************
                 walkTree(root);  Inorder walk on the tree
                 Nodei=0;
                ****/
            }
            count_des = 0;
            cout<<"frame"<<counter<<"\n"<<endl;
            int j;
            #pragma omp parallel shared(desc,desc_show2,descriptors,root,count_des) private(visited,found,best_dist,temp)
            { 
                #pragma omp for schedule(dynamic) nowait 
                for (j = 0; j< descriptors.rows; ++j)
                {
                    for (i = 0; i < 32; ++i)
                    {   
                        testNode.x[i] = desc.at<float>(j,i);
                    }
                    visited=0;
                    found=0;
                    nearest(root, &testNode, 0, MAX_DIM, &found, &best_dist);
                    if (sqrt(best_dist) < threshold)
                    {   
                        //cout<<"best_dist:"<<sqrt(best_dist)<<"\n";
                        #pragma omp critical
                        for(i=0;i<32;i++)
                        {
                                temp = desc_show2.at<float>(j,i);
                                desc_show2.at<float>(count_des,i) = temp; //count_des will always be less than j 
                        }

                        // float xcoord = kp[j].pt.x;
                        // cout<<"keypoint="<<xcoord<<"\n";
                        #pragma omp critical
                        count_des++; 
                    }
                    // cout<<"Number of matched descriptors ="<<count_des<<"\n";
                }
            }
            counter++;
            // cout<<counter;
            descriptors.release();
            desc.release();
            double run_time = omp_get_wtime() - start_time;
            cout << "The run took: " << run_time*1000000/count_des << " microsecs" << endl;
        }
        // double run_time = omp_get_wtime() - start_time;
        // cout << "The run took: " << run_time*1000 << " msecs" << endl;
        drawKeypoints(gray, kp, gray);
        imshow("gray", gray);
        if(waitKey(30) >= 0) break;
    }
    // }
    return 0;
}
