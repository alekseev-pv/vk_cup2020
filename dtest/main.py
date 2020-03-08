import sys

import pandas as pd
import numpy as np

import joblib

from scipy.sparse import coo_matrix, hstack, vstack

from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.ensemble import RandomForestRegressor


def load_tasks(tasks_filename):
    return pd.read_csv(tasks_filename, sep="\t")


#### mean and std from validate_answers

# 	   at_least_one 	at_least_two 	at_least_three
# mean 	0.115441 	0.065805 	0.047170
# std 	0.146146 	0.117812 	0.099029

at_least_one_mean = 0.115441
at_least_two_mean = 0.065805
at_least_three_mean = 0.047170

# from validate.tsv

mean_cpm = 162.425595

mean_audience_size = 1090.087302 	

mean_hours = 101.178571

# all views on 0..23 hours
hours_views = [18768, 13176, 8280, 7892, 10208, 14552, 21520, 28536, 31396, 32320, 33056, 34084, 33696, 34716, 33924, 34344, 34364, 35404, 37000, 37164, 39116, 37388, 34324, 25000, 21056, 14468, 8820, 8532, 10344, 14048, 21616, 28516, 31508, 33012, 33420, 33692, 33432, 32832, 33932, 34628, 34344, 34912, 37088, 38496, 40408, 39520, 36192, 27092, 21648, 13992, 9100, 8388, 10204, 14308, 21896, 29108, 32452, 35312, 34356, 33980, 34308, 34912, 34876, 34836, 35896, 35564, 37124, 39212, 39660, 40320, 36328, 27888, 21012, 14732, 9284, 8176, 10156, 14460, 21108, 28864, 32772, 33952, 34240, 34784, 34816, 35092, 34280, 33408, 35580, 36716, 36968, 37772, 39836, 40568, 36120, 26936, 20564, 14432, 9124, 8316, 10308, 14132, 21724, 28704, 33036, 35008, 35324, 35572, 34840, 35236, 34664, 34516, 34764, 35652, 35460, 36300, 36736, 37284, 34564, 27464, 22168, 15368, 9764, 7916, 9032, 12004, 17372, 25152, 29012, 32692, 32668, 31848, 30408, 30164, 30216, 29796, 30032, 29216, 31360, 32612, 32724, 34224, 31556, 26208, 23664, 16924, 9984, 7012, 7388, 9352, 14388, 19652, 25120, 28196, 29584, 28400, 28032, 27228, 27916, 26820, 27716, 27388, 29180, 29376, 29860, 31112, 29116, 21348]

mean_views = np.mean(hours_views) * mean_hours

mean_views = 2765203.1183892144

# make list with count of every hour on 24 cycle
def make_hours(begin,delta):
    r=np.empty(24);
    end = begin + delta
    
    begin1 = begin % 24
    
    counts = (delta - (24 - begin1)) // 24
    tail = (delta - (24 - begin1)) % 24
    
    r.fill(counts)
    for i in range(begin1, 24):
        r[i] = r[i] + 1
    for i in range(0, tail):
        r[i] = r[i] + 1
    return r

def main():
    tests_filename = sys.argv[1]

    tasks = load_tasks(tests_filename)
  
    
    tasks = tasks.assign(
        at_least_one= at_least_one_mean , #* (tasks['hour_end']-tasks['hour_start'])/mean_hours,
        at_least_two= at_least_two_mean , # * (tasks['hour_end']-tasks['hour_start'])/mean_hours,
        at_least_three= at_least_three_mean, # * (tasks['hour_end']-tasks['hour_start'])/mean_hours,
        hour = tasks['hour_end']-tasks['hour_start']+1,
        age=0,
        age_min=0,
        age_max=0,
        cpmMinMin=0,
        cpmMinMea=0,
        cpmMaxMax=0,
        cpmMaxMea=0,
        cpmMaxMaxi=0,
        CPMxHour=0,
        CPMxHour1=0,
        CPMxHour2=0,
        CPMxHour3=0,
        CPMxHour4=0,
    )
    

    # make some magic with time, make 24 x 7 cycle and use first and last hour
    tasks['hour_start'] = tasks['hour_start'] % (24*7*4)
    tasks['hour_end'] =  tasks['hour_start'] + tasks['hour']-1
   
    #tasks['hour24'] = tasks['hour_start'] % 24
   
    tasks['publishers_size'] = 0
    tasks['views_sum'] = 0

    tasks['publishers_size'] = tasks['publishers_size'].astype('UInt8')
    tasks['views_sum'] = tasks['views_sum'].astype('int')
     
    # count of publishers and hours views
    for i in range(len(tasks.index)):
        tasks.loc[i,'publishers_size'] = int(len(tasks['publishers'][i].split(',')))
        
        t = 0
        for h in range(int(tasks.iloc[i]['hour_start']),int(tasks.iloc[i]['hour_end'])+1):
            t = t + hours_views[h %(24*7)]      
            
        tasks.loc[i,'views_sum'] = t
        
    # load some stats
    
    usersT = np.load('usersT.npy')
    
    usersH = np.load('usersH.npy')
 
    
    usersCPMmin = np.load('usersCPMmin.npy')
    usersCPMmax = np.load('usersCPMmax.npy')
    
    # 24 hours cycle
    
    usersHours24 = np.load('usersHours24.npy', allow_pickle=True)
    usersCPM24 = np.load('usersCPM24.npy', allow_pickle=True)

    usersViews24 = np.load('usersViews24.npy', allow_pickle=True)    
    
    
    age_bins = list()
    abins = [0, 13, 15, 18, 19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 50, 55, 60, 65, 70, 95, 140]
 
    for i in range(len(tasks.index)):
        
        # how many count of every hour in cycle?
        r_hours = make_hours(tasks.loc[i,'hour_start'], tasks.loc[i,'hour'])
            
        u_list = list(map(int, tasks['user_ids'][i].split(',')))
        
        tasks.loc[i,'sex'] = usersT[1][u_list].mean()
        
        age_array = usersT[2][u_list]

        age_bins.append(np.histogram(age_array, bins=abins)[0] * 100.0 / len(age_array))
        
        tasks.loc[i,'age'] = age_array.mean()
        tasks.loc[i,'age_min'] = age_array.min()
        tasks.loc[i,'age_max'] = age_array.max()
        
        tasks.loc[i,'town'] = usersT[3][u_list].mean() 
        
        publishes_ids = list(map(int,tasks['publishers'][i].split(',')))
        
        probable_views = 0
        #probable_views1 = 0
        
        cpmMinMin = 9999999
        #cpmMinMea = 0

        cpmMaxMax = -1
        #cpmMaxMea = 0
        
        # magic stats
        CPMxHour = 0
        CPMxHour1 = 0
        CPMxHour2 = 0
        CPMxHour3 = 0
        CPMxHour4 = 0
        
        for uid in map(int, tasks.iloc[i]['user_ids'].split(',')):

            probable_views = probable_views + usersH[uid][publishes_ids].sum()
            #probable_views1 = probable_views1 + usersH1[uid][publishes_ids].sum()
            
            ''''''
            ### CPMmin
            t = usersCPMmin[uid][publishes_ids]      
            tmin = t.min()
      
            if (cpmMinMin > tmin):
                cpmMinMin = tmin
            
            #cpmMinMea = cpmMinMea + t.sum()
            
            ### CPMmax 
            t = usersCPMmax[uid][publishes_ids]      
            tmax = t.max()
      
            if (cpmMaxMax < tmax):
                cpmMaxMax = tmax
            
            #cpmMaxMea = cpmMaxMea + t.sum()
            
            local_CPM = tasks.loc[i,'cpm']
            
            for pp in publishes_ids:
                for t in usersHours24[uid][pp]:
                    cur_CPM = usersCPM24[uid][pp][t]  #  * n / n_all
                    
                    if (cur_CPM > 0):
                        if (cur_CPM < local_CPM):
                            CPMxHour = CPMxHour + 1 
                            CPMxHour1 = CPMxHour1 + 1                        
                            CPMxHour2 = CPMxHour2 + 1
                            CPMxHour3 = CPMxHour3 + r_hours[t] # ex. in 13 hours we have 1 view, and for 3 days we have it 3 times
                            CPMxHour4 = CPMxHour4 + usersViews24[uid][pp][t]
                            
                        elif (cur_CPM == local_CPM):
                            #CPMxHour = CPMxHour + 1
                            CPMxHour1 = CPMxHour1 + 0.5                        
                            CPMxHour2 = CPMxHour2 + 1
                            CPMxHour3 = CPMxHour3 + 0.5 * r_hours[t]   
                            CPMxHour4 = CPMxHour4 + 0.5 * usersViews24[uid][pp][t]
                        else:                     
                            CPMxHour2 = CPMxHour2 + local_CPM / cur_CPM                     
                            CPMxHour3 = CPMxHour3 +  r_hours[t] * local_CPM / (2 * cur_CPM)   
                            CPMxHour4 = CPMxHour4 + usersViews24[uid][pp][t] * local_CPM / (2 * cur_CPM)
                        
        tasks.loc[i,'p_history'] = probable_views / (tasks.loc[i,'publishers_size'] * tasks.iloc[i]['audience_size'])
        #tasks.loc[i,'p_history1'] = probable_views1 / (tasks.loc[i,'publishers_size'] * tasks.iloc[i]['audience_size'])

        
        tasks.loc[i,'cpmMinMin'] = cpmMinMin / local_CPM # / ??? / validate.loc[i,'cpm'] ???
        #tasks.loc[i,'cpmMinMea'] = cpmMinMea / probable_views1 / tasks.loc[i,'cpm']# / ??? / validate.loc[i,'cpm'] ???

        tasks.loc[i,'cpmMaxMax'] = cpmMaxMax / local_CPM # / ??? / validate.loc[i,'cpm'] ???
        tasks.loc[i,'cpmMaxMaxi'] = local_CPM / cpmMaxMax 
        
        #tasks.loc[i,'cpmMaxMea'] = cpmMaxMea / probable_views1 / tasks.loc[i,'cpm']# / ??? / validate.loc[i,'cpm'] ???

        ''' '''
        tasks.loc[i,'CPMxHour'] = CPMxHour
        tasks.loc[i,'CPMxHour1'] = CPMxHour1
        tasks.loc[i,'CPMxHour2'] = CPMxHour2
        tasks.loc[i,'CPMxHour3'] = CPMxHour3   
        tasks.loc[i,'CPMxHour4'] = CPMxHour4   
        
    tasks['cpmAVE'] = (tasks['cpmMaxMax'] + tasks['cpmMinMin'])  / 2
    
    ##
    # users and publishers dictionary load
    v1 = joblib.load('v1.pkl') #CountVectorizer(token_pattern='[0-9]+', dtype='UInt8')
    v2 = joblib.load('v2.pkl') #CountVectorizer(token_pattern='[0-9]+', dtype='UInt8')
    
    p_ids = v1.transform(tasks['publishers'])
    
    u_ids = v2.transform(tasks['user_ids'])

    # base features
    features = ['cpm', 'hour_start', 'hour_end','audience_size','hour','publishers_size', 'views_sum','sex','age','age_min','age_max','town','p_history', 'cpmMinMin', 'cpmMaxMax', 'cpmMaxMaxi','cpmAVE','CPMxHour','CPMxHour1','CPMxHour2', 'CPMxHour3', 'CPMxHour4']
      
    trainX = tasks[features].values
    
    # train(features + publishers vector + user_ids vector + age_bins)
    trainX = np.hstack((trainX, p_ids.toarray(),u_ids.toarray(),np.array(age_bins)))
    
    # load regressors
    r1 = joblib.load('r1.pkl')
    r2 = joblib.load('r2.pkl')
    r3 = joblib.load('r3.pkl')


    tasks['at_least_one'] = r1.predict(trainX)
    tasks['at_least_two'] = r2.predict(trainX)
    tasks['at_least_three'] = r3.predict(trainX)
    
    ''' # make integers in a/b  '''
    for i in range(len(tasks.index)):
        tasks.loc[i,'at_least_one'] = int(0.5 + tasks.loc[i,'audience_size'] * tasks.loc[i,'at_least_one']) / tasks.loc[i,'audience_size']
        tasks.loc[i,'at_least_two'] = int(0.5 + tasks.loc[i,'audience_size'] * tasks.loc[i,'at_least_two']) / tasks.loc[i,'audience_size']
        tasks.loc[i,'at_least_three'] = int(0.5 + tasks.loc[i,'audience_size'] * tasks.loc[i,'at_least_three']) / tasks.loc[i,'audience_size']
    
    
    tasks[['at_least_one', 'at_least_two', 'at_least_three']].to_csv(sys.stdout, sep="\t", index=False, header=True)


if __name__ == '__main__':
    main()
