#include <cstdlib>
#include <string>
#include <cmath>
#include <vector>
#include <memory.h>
#include <algorithm>
//#include "/home/zfy/MATLAB/extern/include/mex.h"
#include "mex.h"
//input:
//    - data, which is a 2-dim matrix.  V
//    - dims: [nrow, ncol, nband]
//    - win_size
//    - percent: 选择百分之多少
//output:
//    - ri
//    - ci
//    - value
// 2012-3-31

void scaleVector(double* src, double* dst, int num);

void mexFunction( int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[] )
{
    //! 检查 输入、输出 参数个是否正确
    if( nrhs != 4 ){
        mexErrMsgTxt(" must input 4 parameters");
    }

    if( nlhs != 3 ){
        mexErrMsgTxt(" must out 3 parameters");
    }

    //! 提取输入参数 0 - 3，总共4个参数
    double* pdat = mxGetPr (prhs[0]);
    double* pdims = mxGetPr (prhs[1]);
    int winSize  = mxGetScalar (prhs[2]);
    double percent = mxGetScalar (prhs[3]);
    percent /= 100;
//    mexPrintf ("percent = %f \n", percent);

    //! 设置各种系数
    int nrow = pdims[0];
    int ncol = pdims[1];
    int nband = pdims[2];
    int nsmp = nrow * ncol;
    int nebSize = (2*winSize + 1) * (2*winSize + 1);
    int nebRealSize = std::ceil(nebSize * percent);
    int num = (nrow-2*winSize) * (ncol-2*winSize) * nebRealSize * 2;

    //! 设置输出参数，并提取指针
    plhs[0] = mxCreateDoubleMatrix (num, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix (num, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix (num, 1, mxREAL);
    double* rid = mxGetPr (plhs[0]);
    double* cid = mxGetPr (plhs[1]);
    double* val = mxGetPr (plhs[2]);

    //! 定义临时储存向量的内存
    double* ndst = new double [nband];
    double* mdst = new double [nband];

    //! 计算局部window里面，相对中心点的位置(单坐标位置)
    std::vector<int> neb_index;
    for(int i=-winSize; i != winSize+1; i++)
        for(int j=-winSize; j != winSize+1; j++)
            neb_index.push_back (j + i*nrow);

    //! 定义需要用到的临时数据
    double *mpdat, *npdat, dist, tol;
    int mindx, nindx, nebNum, sparseNum = 0;

    std::vector<int>  indxVec;
    std::vector<double> valVec;

    for(int j=winSize; j != ncol-winSize; j++) {
        for(int i= winSize; i != nrow-winSize; i++) {
            //! 指向矩阵的第mindx列，去均值、归一化该列
            mindx = i  + j*nrow;
            mpdat = pdat + mindx*nband;
            scaleVector (mpdat, mdst, nband);

            //! 以下是该像元对应的局部邻域内的像元的操作
            //! @keyword: 计算邻域中心点和同类点的相关系数
            for(int ni=0; ni != nebSize; ni++) {
                nindx = mindx + neb_index[ni];
                //! 如果邻域内的像元和中心像元属于同一类，则进入if里面
                indxVec.push_back (nindx);
                npdat = pdat + nindx*nband;
                scaleVector (npdat, ndst, nband);
                dist = 0;
                for(int ch=0; ch != nband; ch++) {
                    dist += mdst[ch] * ndst[ch];
                }
                valVec.push_back (dist);
            }
            nebNum = std::ceil (percent*valVec.size ());
            if(nebNum != 0) {
                std::vector<double> sortValVec(valVec);
                std::sort(sortValVec.begin (), sortValVec.end ());
                tol = sortValVec[sortValVec.size () - nebNum];
                for(int vi = 0; vi != valVec.size ();  vi++) {
                    if(valVec[vi] >= std::max(tol, 0.7)) {
                        rid[sparseNum] = mindx;
                        cid[sparseNum] = indxVec[vi];
                        val[sparseNum++] = valVec[vi];

                        rid[sparseNum] = indxVec[vi];
                        cid[sparseNum] = mindx;
                        val[sparseNum++] = valVec[vi];
                    }
                }
                valVec.clear ();
                indxVec.clear ();
                sortValVec.clear ();
            }
        }
    }
    delete [] ndst;
    delete [] mdst;
}

void scaleVector(double* src, double* dst, int num) {
    double normS = 0;
    int i;
    for(i=0; i != num; i++) {
        normS += std::pow (src[i], 2);
    }
    normS = std::pow (normS, 0.5);
    for(i=0; i != num; i++) {
        dst[i] = src[i] / normS;
    }
}


