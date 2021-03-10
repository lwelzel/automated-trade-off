#These functinos were written by Nathaniel Peter Stebbins Dahl during the Save the Earth from Asteroids

import numpy as np

def boundary_maker(name, scores, hb, boundary_condition, boundaries):
    '''
    The boundary maker function is the lead up to the map score function. The map score function maps a score on the x axis that can be any number to a score on the y axis between 0 and 1. Where 1 is the best.
    Sometimes you might want to set specific boundaries on the x axis. 
    For example, Anything over budget aka the money score = greater then your budget you want to give a score of 0. You also don't want your product to be too cheap. So your budget is 40 euros but anything below 20 shouldnt be getting extra score.
    You can then use the fixed boundary conditions at 20, 40 and set hb = 0. This means any design with a money score above 40 will be given 0 points and any score below 20 will only get 1 point. 

    You can also use the min and max of the input scores or x number of standard deviations from the scores.
    '''

    #TODO ADD exception handling

    #All we do is chech what the input for type of boundary condition it is and then set it. Pretty self explanatory
    if boundary_condition == "MinMax Scores":
        lb, ub  = min(scores),max(scores)

    elif boundary_condition == "Fixed Range":
        boundaries = list(boundaries.split(','))   
        if len(boundaries) != 2:
            raise Exception("Its broken! You wanted a fixed range but I cannot see the lower bound or upper bound. Avoid  using spaces and only put a , in between!")
        lb, ub  = boundaries[0],boundaries[1]

    elif boundary_condition == "STD Scores":
        stnd_d      = np.std(scores)
        ave         = np.average(scores)
        boundaries  = float(boundaries)
        if type(boundaries) != int and type(boundaries) != float:
            raise Exception("Oi its broken, the expected boundary was int or float but it is "+str(type(boundaries))+" The error occured while looking at "+ name)
        lb, ub  = ave - boundaries*stnd_d, ave + boundaries*stnd_d

    else:
        raise Exception("No boundary set?!?! wtf")

    #As we don't do anything with hb here, we need to ensure that the lower boundary is actually lower then the upper boundary
    if ub<=lb:
        raise Exception("Upper Bound smaller then lower bound.")
    return [lb,ub,int(hb)]

def map_score(score, worst_and_best_score, analysis_function , scale_p):
    '''
    To play around with what the mapping functions actually do check out this link: www.desmos.com/calculator/justratyk4
    We map a function from any number real (positive) to a range between 0 and 1
    You can use the different mapping functions to spread out of group different parts of your input domain
    The parameter p is important. it adjusts the shape of the curves. Again look at the desmos link.

    For the increasing and decreasing rts. Keep p between 0 and 2. Values close to 0 cause extreme curves. Larger values of p make the curves look like the linear mapping. At 2 the curve is already very gentle

    For the S curve. Best results will be found between 2 and 5. 3 or 4 is a good starting point. Anything below 1 and the function really does not work and above 5 it becomes very extreme.
    Here again I really recommend plotting the functions or clicking the link!
    '''
    #We will define the best score possible as 1 and the lowest as 0
    #Thus everything we do here is a mapping from R to [0,1]
    lb, ub, hb  = float(worst_and_best_score[0]), float(worst_and_best_score[1]), worst_and_best_score[2]
    
    #Higher is Better
    if hb:
        #Are we better or worse then boundaries?
        if score <= lb:
            return 0
        elif score >= ub:
            return 1

        #Given we are inside the boundaries we now check what type function is expected
        if analysis_function == "Linear Scaling":
            #Do the thing
            return (score - lb)/(ub-lb)

        elif analysis_function == "Increasing RTS":
            #Do the next thing
            return (1-np.exp((score-lb)/scale_p))/(1-np.exp((ub-lb)/scale_p))

        elif analysis_function == "Decreasing RTS":
            return (1-np.exp(-(score-lb)/scale_p))/(1-np.exp(-(ub-lb)/scale_p))

        elif analysis_function == "S Scaling":
            #For a logistics S curve we have x0 (mid point) (inflextion point of the S)
            x0 = (ub-abs(lb))/2
            #L, curve max value (1)
            L = 1
            #k, steepness/growth rate. For this we will just use the variable scale_p
            return L/( 1+np.exp( -scale_p*(score-x0) ) )
        else:
            raise Exception("Error in analysis function type")           
    
    #Lower is Better
    elif not hb:
        #Are we better or worse then boundaries?
        if score <= lb:
            return 1
        elif score >= ub:
            return 0
        
        #Given we are inside the boundaries we now check what type function is expected
        if analysis_function == "Linear Scaling":
            #Do the thing
            return (ub-score)/(ub-lb)

        elif analysis_function == "Increasing RTS":
            #Do the next thing
            return (1-np.exp((ub-score)/scale_p))/(1-np.exp((ub-lb)/scale_p))

        elif analysis_function == "Decreasing RTS":
            return (1-np.exp(-(ub-score)/scale_p))/(1-np.exp(-(ub-lb)/scale_p))

        elif analysis_function == "S Scaling":
            #For a logistics S curve we have x0 (mid point) (inflextion point of the S)
            x0 = (ub-abs(lb))/2
            #L, curve max value (1)
            L = 1
            #k, steepness/growth rate. For this we will just use the variable scale_p
            return L/( 1+np.exp( scale_p*(score-x0) ) )
        else:
            raise Exception("Error in analysis function type")
    
def zlist(n):
    ''' Returns a list of zeros n long'''
    return [0]*n

def score_to_fraction(weights):
    '''Basically a normalizing function.'''
    total = 0
    realweights = []
    for w in range(len(weights)-1):
        total += float(weights[w+1])
        realweights.append(float(weights[w+1]))

    realweights[:] = [x / total for x in realweights]
    return realweights,total



#Create a graphic test for these. Easy to check the mapping that way
#Check the S curve