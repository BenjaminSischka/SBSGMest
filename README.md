# SBSGMest
EM-Based Estimation Routine for the Stochastic Block Smooth Graphon Model

[Sischka, B. & Kauermann, G. (2022). Stochastic Block Smooth Graphon Model. arXiv preprint arXiv:2203.13304]


 - SBSGMpy:
 -------------

	Python package for conducting the EM-based estimation routine to estimate Stochastic Block Smooth Graphon Models

	* dependencies [(~ ...) = included in python standard library]:
		(~ sys)
		(~ os)
		(~ io)
		(~ copy)
		(~ warnings)
		(~ operator)
		(~ csv)
		(~ pickle)
		~ matplotlib
		~ numpy
		~ scipy
		~ math
		~ cvxopt
		~ networkx
		~ sklearn
		~ mpl_toolkits



 - application.py:
 -----------------

	-> File to execute for running the algorithm.

	 * Settings for manual adaptation are clearly delimited.

	 * For applying the alliance network:
	     1. download data set alliances_strong_post_adjMat_2016.csv
	     2. embed file in folder structure Data/alliances/***.csv
	     3. place Data folder at same level as package folder SBSGMpy

	 * Command in terminal / shell / console to run the script:
	   cd ~/>>path_to_folder<</SBSGMest
	   python3 application.py > output.txt

	 * Adjustment when running in interactive session:
	   dir1_ = os.path.dirname(os.path.realpath(''))

	 * Creates new local folder named 'Graphics' where all figures will be saved.



 - Data:
 -------
  
  Simulation data -- configurations (see application.py)
  
  * Assortative Structure with Smooth Within-Group Differences:
    idX = 1
    [ -> byExID2(idX=205)]
  * Core-Periphery Structure:
    idX = 2
    [ -> byExID2(idX=207)]
  * Mixture of Assortative and Disassortative Structures under Equal Overall Attractiveness:
    idX = 3
    [ -> byExID2(idX=208)]
  
  
	Data sources which have been used for real-world network examples

	* political blog network, downloaded from 

			http://www-personal.umich.edu/~mejn/netdata/ [online; accessed 08-14-2020].
			[personal website of Mark Newman]
			
			and assembled by
			
			[Adamic, L. A. and Glance, N. (2005). The political blogosphere and the 2004 US 
			election: divided they blog. In Proceedings of the 3rd international workshop on 
			Link discovery (pp. 36-43).]


	* military alliance network (processed), downloaded from and assembled by

			http://www.atopdata.org/data [online; accessed 03-02-2020].
			[Leeds, B. A., J. M. Ritter, S. McLaughlin Mitchell and A. G. Long (2002). Alliance 
			Treaty Obligations and Provisions, 1815-1944. International Interactions 28: 237-260.]


	* human brain functional coactivation network, downloaded from

			https://sites.google.com/site/bctnet/datasets [online; accessed 10-15-2019].
			[Rubinov, M. and Sporns, O. (2010). Brain Connectivity Toolbox (MATLAB).]

			and assembled by

			[Crossley, N. A., Mechelli, A., Vértes, P. E., Winton-Brown, T. T., Patel, A. X., 
			Ginestet, C. E., McGuire, P. and Bullmore, E. T. (2013). Cognitive relevance of the 
			community structure of the human brain functional coactivation network. Proceedings 
			of the National Academy of Sciences 110 11583–11588.]
