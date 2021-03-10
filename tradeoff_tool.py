import pandas as pd
import numpy as np
from Stat_Tools import boundary_maker, map_score, zlist, score_to_fraction
import copy as cp

class trade_off:
    def __init__(self,criteria_list,design_list):
        self.criteria       = criteria_list
        self.designs        = design_list
        
        #normalize weights
        normalized,self.weight_total = score_to_fraction(self.criteria.loc[0,:].to_numpy())
        j = 0
        crit_names = self.criteria.columns.tolist()
        crit_names.pop(0)
        for crit in crit_names:
            self.criteria.loc[0,crit] = normalized[j]
            j+=1

        #Adding a column for Scoring
        table_shape         = np.zeros((self.designs.shape[0],self.designs.shape[1]+2))
        table_columns       = self.designs.columns.tolist()
        for column in range(1,len(table_columns)):
            table_columns[column] = table_columns[column] + ": \n"+ str(round(100.*normalized[column-1],1)) + "%"
        table_columns.append("Total Score")
        table_columns.append("Confidence [%]")

        #Initialize Output DF
        self.output_data   = pd.DataFrame(table_shape, columns = table_columns)
        for designs in range(len(self.designs['Design Options'].tolist())):
            self.output_data.loc[designs, 'Design Options'] = self.designs['Design Options'].tolist()[designs]

        #run the trade off
        self.run_to()
        
    def run_to(self):
        ''' This function takes the inputs for criteria weight, design option score for criteria and some others. It first finds the scoring boundary conditions. 
        It then loops through the designs and evaulates each option. Finally it adds the total score'''

        #Read some local variables for easier calling.
        crit_names = self.criteria.columns.tolist()
        w          = self.criteria.iloc[0]
        hb         = self.criteria.iloc[1]
        anal_func  = self.criteria.iloc[2]
        scale_p    = self.criteria.iloc[3]
        bound_con  = self.criteria.iloc[4]
        bounds     = self.criteria.iloc[5]

        #run boundary code
        #First we initialize the boundary condition row and concat it onto the bottom of the criteria datafram
        lb_ub_hb_row = pd.DataFrame( [['lb_ub_hb']+zlist(self.criteria.shape[1]-1)], columns=crit_names)
        self.criteria = pd.concat( [self.criteria, lb_ub_hb_row],ignore_index=True)

        #We then analyze the data given and create boundaries for the scoring function.
        #Any score below the lb will be given a final score of 0. Any score above  the ub (upper bound) will be given a final score of 1. 
        #Basically we are normalizing the scoring between a lower boundary and an upper boundary. 
        #This is of course true for Higher is better. When the data is lower is better the reverse is true. (lower then lower bound returns 1)
        #Again though. This section only comes up with the lower and upper bound. It also adds the higher better true false for ease of use later.
        i = 1
        crit_names.pop(0)
        for crit in crit_names:
            lb_ub_hb    = boundary_maker(crit, self.designs[crit].tolist(), hb[crit], bound_con[crit], bounds[crit])
            self.criteria.iloc[8,i] = lb_ub_hb
            i += 1

        #Now we will want to iterate through the designs and see how well they do given how well they scored on each individual criteria
        #TODO Multicore for looping through the designs? they do not have to be dependent. Probably a pain to implement tho, easier at a higher level
        #We first read a list
        dess = self.designs['Design Options'].tolist()
        
        #We then iterate through each design. Maybe its possible to due this fully using arrays but time is money.
        for design in range(len(dess)):
            #We need a total score for the current design
            total_score = 0.
            #We now iterate over each criteria. I was bad at this point so needed the crutch of cr = 1 to avoid calling the row header. Later I do this more elegantly
            cr = 1
            for crit in crit_names:
                #We read the technical score at the current criteria and current design
                #This gets passed into the mapping function. The mapping function returns a value between and including 0 and 1. 1 being best and 0 being worst. No matter the state of hb.
                #Finally we add it to the total score of this design
                l_score     = self.designs.loc[design,crit]
                m_score     = map_score(l_score, self.criteria[crit].iloc[8], anal_func[crit], float(scale_p[crit]))
                total_score += m_score*float(w[crit])

                #We save some of the data for use in outputting into the tables
                if crit == 'System Efficiency [1E-5]': #Stop gap solution for this specific TO
                    l_score *= 1E5
                self.output_data.iloc[design,cr] = f'{l_score:.3f} -> {m_score:.3f}'#str([round(l_score,), round(m_score,3)])
                cr += 1

            #Finally we append the score of the design.
            self.output_data.iloc[design,-2] = f'{total_score:.3f}'



class sens_anal:
    '''Here we shuffle all the variables according to specified normal distributions. We then see in what % of cases the outcome of the tradeoff changes. Make sure to keep n very large'''
    def __init__(self,trade_off,sens_data, n = 100, rando_crit = True, rando_score = True, rando_p = True):
        self.trade_off  = trade_off
        self.og_to      = cp.deepcopy(trade_off)
        self.sens_lst   = np.zeros((len(trade_off.designs['Design Options'].tolist()),n))
        self.sens_data  = sens_data
        self.n          = n
        self.rando_crit = rando_crit
        self.rando_score= rando_score
        self.rando_p    = rando_p
        
        #the criteria stored in trade_off have already been normalized. We need the 'true' values, hence the _t
        crits           = self.og_to.criteria.iloc[0].to_numpy()
        self.crits      = np.delete(crits,0).astype(float,copy=False)  
        self.crits_n    = len(self.crits)

        self.run_sens()

    def run_sens(self):
        '''Basics of how this works is we first take create an array with all inputs x n iterations. We then add random noise following a normal distribution.'''
        #creates a large tiled list of the criterian weights.
        #Each ROW is the same at this point 

        big_crits   = np.tile(self.crits,[self.n,1]).astype(float,copy = False) # need to do this to ensure we can later do the add operation
        #Now ONLY if we want  to randomize weights add scaled normal distribution
        if self.rando_crit:
            if int(sum(self.crits)) != 1:
                raise Exception("Something went wrong! sum of normalized criteria is not 1")
            #The standard deviation for the normal distribution is given by input*criteria (should already be normalized)
            #This means the input variable is a fractional percentage. So it should be below one.
            rando_std_crits = np.tile(np.multiply(self.og_to.criteria.iloc[6,1:].to_numpy().astype(float,copy = False),self.crits),[self.n,1])
            big_crits       += np.random.normal(0,rando_std_crits,(self.n,self.crits_n))
            #Notice the STD is normalized as well.
            #TODO Need to deal with below 0 and above 1

        #now we do the same for the others
        score_table = self.og_to.designs.iloc[0:len(self.og_to.designs['Design Options'].tolist()),1:self.crits_n+1].to_numpy()
        big_scores  = np.repeat(score_table[:,:,np.newaxis],self.n,axis=2).astype(float,copy = False)
        if self.rando_score:
            #Same as above. input should be fractional percentage to make the std. Just now its not normalized
            rando_std_scores    = np.repeat(np.multiply(self.og_to.criteria.iloc[7,1:].to_numpy().astype(float,copy = False),score_table)[:,:,np.newaxis],self.n,axis=2)
            big_scores          += np.random.normal(0,rando_std_scores,big_scores.shape)
            #TODO agian deal with negative numbers?
        
        #First read the list of p values
        p_table     = self.og_to.criteria.iloc[3].to_numpy()
        #clean up list
        p_table     = np.delete(p_table,0)
        #Create big list
        big_p       = np.tile(p_table,[self.n,1]).astype(float,copy = False)

        if self.rando_p:
            #Standard deviation will be defined by inputs from sens_data file. again it will be multiplied with the p value so should be
            #considered a fractional percentage
            rando_std_p = np.zeros(len(self.og_to.criteria.iloc[3,1:].to_numpy()))
            #cleaner ways...
            for i in range(len(self.og_to.criteria.iloc[3,1:].to_numpy())):
                if self.og_to.criteria.iloc[2,i+1] == 'Linear Scaling':
                    rando_std_p[i] = float(self.sens_data['Linear Scaling'])

                elif self.og_to.criteria.iloc[2,i+1] == 'Increasing RTS':
                    rando_std_p[i] = float(self.sens_data['Increasing RTS'])

                elif self.og_to.criteria.iloc[2,i+1] == 'Decreasing RTS':
                    rando_std_p[i] = float(self.sens_data['Decreasing RTS'])

                elif self.og_to.criteria.iloc[2,i+1] == 'S Scaling':
                    rando_std_p[i] = float(self.sens_data['S Scaling'])

                else:
                    raise Exception("Something went wrong! Somehow invalid analysis function.")
                
            rando_std_p = np.tile(np.multiply(rando_std_p,p_table.astype(float,copy = False)),[self.n,1])
            big_p       += np.random.normal(0,rando_std_p,big_p.shape) #;)
        
        #should multicore the following for loop. Break up the list. As long as you write to the correct place.
        #Could also have the code write to seperate arrays if that is faster. idk 

        #Deep copy is a seperate object. We will change these for each n
        deep_crits  = self.og_to.criteria.copy() #default is a deep copy
        deep_dess   = self.og_to.designs.copy()
        #We need this to read out the indicies and columns
        
        #We now have big_p, big_scores, big_crits all randomized
        #So we write the adjusted input values for each n into the deep copies
        for i in range(self.n):
            #awkwardly we need this to all be dataframes oof.
            #Adjust weights for random noise:
            deep_crits.iloc[0,1:]   = big_crits[i,:]
            deep_crits.iloc[3,1:]   = big_p[i,:]
            deep_dess.iloc[:,1:]    = big_scores[:,:,i]

            #run the current random trade_off
            deep_to = trade_off(deep_crits,deep_dess)

            #Put a 1 in every position where the design option was the best option.
            #This will result in a len(designs) by n array with 1s and 0s. There should only be one 1 in a column. 
            self.sens_lst[np.where(deep_to.output_data.iloc[:,-2] == np.amax(deep_to.output_data.iloc[:,-2])),i] = 1
        
        
        self.sensativity = np.sum(self.sens_lst,axis=1)/self.n*100
        for i in range(len(self.sensativity)):
            self.trade_off.output_data.iloc[i,-1] = f'{self.sensativity[i]:.3f}'
        

