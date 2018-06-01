Project: the comparison of the three actor-critic reinforcement learning methods

author: Feiyun Zhu

email:  fyzhu0915@gmail.com


---------------------------------------------------------------------------------------------------------
NOTE THAT:

the file starts with r_Perf_vs_* can be run without changing the code.

---------------------------------------------------------------------------------------------------------
the structure of the files and folders

codeShare_BatchRL
	|-readMe.txt
	|-ActCribic_fyzhu_v8.pdf is a draft to introduce the actor-critic RL methods (i.e. algorithms), the data generaation model and the feature construction.
	|-Data_RL_algs\ is a folder contains all the functions we created, including 1) the algorithms to generate the trajectory, 2) the RL learning method, 3) the evaluation methods for the estimated policy. 
	|-r_Perf_vs_alpha.m  	the matlab script that runs the performance vs. alpha
	|-r_Perf_vs_Iterations  the matlab script that runs the performance vs. Iterations
	|-r_Perf_vs_NoiseInData.m the matlab script that runs the performance vs. noise in data
	|-r_Perf_vs_NPeo.m  the matlab script that runs the performance vs. number of people involved in the experiment
	|-r_Perf_vs_OrderPolynomail.m the matlab script that runs the performance vs. order of RBF basic function
	|-r_Perf_vs_OrderRBF.m  the matlab script that runs the performance vs. order of polynomial basic function
	|-r_Perf_vs_T.m  the matlab script that runs the performance vs. the trajectory length
	|-r_Perf_vs_treatmentFagtigue.m  the matlab script that runs the performance vs. the treatment fagtigue

	
