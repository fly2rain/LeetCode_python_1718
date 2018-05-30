#include <cstdlib>
#include <cmath>
#include <vector>
//#include "/home/zfy/MATLAB/extern/include/mex.h"
#include "mex.h"
//input:
//    - data, which is a 2-dim matrix.  V
//    - dims: [nrow, ncol, nband]
//    - win_size
//    - sigma
//output:
//    - ri
//    - ci
//    - value
//  written by Zhu Feiyun
//  2012-7-18
//  version 2

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

    //! 提取输入参数 0 - 2，总共3个参数
    double* pdat    = mxGetPr (prhs[0]);
    double* pdims   = mxGetPr (prhs[1]);
    int     winSize = mxGetScalar (prhs[2]);
    double  sigma   = mxGetScalar (prhs[3]);

    //! 设置各种系数
    int nrow = pdims[0];
    int ncol = pdims[1];
    int nband = pdims[2];
//    mexPrintf ("nrow= %d\t ncol=%d\t nband=%d \t sigma=%f\n",nrow,ncol,nband,sigma);
    int nebSize = (2*winSize + 1) * (2*winSize + 1);
    int num = (nrow-2*winSize) * (ncol-2*winSize) * nebSize * 2;

//    for(int i=0; i != nband; i++)
//        mexPrintf ("%f\n", (pdat+2*nband)[i]);

    //! 设置输出参数，并提取指针
    plhs[0] = mxCreateDoubleMatrix (num, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix (num, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix (num, 1, mxREAL);

    double* rid = mxGetPr (plhs[0]);
    double* cid = mxGetPr (plhs[1]);
    double* val = mxGetPr (plhs[2]);

    //! 计算局部window里面，相对中心点的位置(单坐标位置)
    std::vector<int> neb_index;
    for(int i=-winSize; i != winSize+1; i++)
        for(int j=-winSize; j != winSize+1; j++)
            neb_index.push_back (j + i*nrow);

    //! 定义需要用到的临时数据
    double *mpdat, *npdat, dist;
    int mindx, nindx, sparseNum = 0;
    for(int j=winSize; j != ncol-winSize; j++) {
        for(int i= winSize; i != nrow-winSize; i++) {
            //! 指向矩阵的第mindx列
            mindx = i  + j*nrow;
            mpdat = pdat + mindx*nband;

            //! 以下是该像元对应的局部邻域内的像元的操作
            //! @keyword: 计算邻域中心点和同类点的相关系数
            for(int ni=0; ni != nebSize; ni++) {
                nindx = mindx + neb_index[ni];
                //! 如果邻域内的像元和中心像元属于同一类，则进入if里面
                npdat = pdat + nindx*nband;
                dist = 0;
                for(int ch=0; ch != nband; ch++) {
                    dist += std::pow (npdat[ch] - mpdat[ch], 2);
                }
//                if (sparseNum < 100)
//                    mexPrintf ("dist= %f\n", dist);
                dist = std::exp (-dist/sigma);

                rid[sparseNum] = mindx;
                cid[sparseNum] = nindx;
                val[sparseNum++] = dist;

                rid[sparseNum] = nindx;
                cid[sparseNum] = mindx;
                val[sparseNum++] = dist;
            }
        }
    }
}

