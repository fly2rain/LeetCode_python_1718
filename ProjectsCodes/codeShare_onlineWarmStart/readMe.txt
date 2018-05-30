Project: the study of the influence of warm start on the online personalized RL method for the mobile health intervention 

author: Feiyun Zhu

email:  fyzhu0915@gmail.com


---------------------------------------------------------------------------------------------------------
NOTE THAT:

the file starts with r_* can be run without changing the code.

---------------------------------------------------------------------------------------------------------
the structure of the files and folders

codeShare_BatchRL
	|-readMe.txt
	|-P5_warmStart_4_persionalIntervetion_v3.pdf is a draft to introduce the three actor-critic RL schemes (i.e. batch, online without warm start, online with warm start), the data generaation model and the feature construction.
	|-P5_warmStart_4_persionalIntervetion_v3.1.tex is the latex source file that is ready to compile.
    |-Data_RL_algs\ is a folder contains all the functions we created, including 1) the algorithms to generate the trajectory, 2) the RL learning method, 3) the evaluation methods for the estimated policy. 
	|-r_compare_3_OL_bandit.m  compare the three actor-critic RL methods in the batch learning setting.
	|-r_SimulateModelDesign.m  is the code script to explore different the continuous simulation data, which could show the two delayed effect: 1) stronger engagement, 2) stronger treatment fatigue. 
	|-r_WS_noWS_eachWS_OL_RL.m  compare the online RL methods that are with warm start and without warm start.
        |-r_noWS_WS_whichAdvantage_OL_RL.m verify which of two improvements really makes a difference. 


	
