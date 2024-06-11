import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# given a set of covariates, run linear regression model for each treatment arm
# take argmax of predicted outcome values to determine treatment assignment

covariates = ["attendance_1", "rating_1", "talktime_tutor_pct_1",
    "spoken_token_tutor_pct_1", "chat_token_tutor_pct_1", "length_utterance_tutor_1",
    "length_utterance_student_1", "length_utterance_tutor_chat_1", "length_utterance_student_chat_1",
    "ratio_students_engaged_1", "normalized_num_turns_1",
    "normalized_num_high_uptakes_1", "normalized_num_eliciting_1", "normalized_num_questions_students_1",
    "normalized_num_questions_tutor_1", "normalized_student_reasoning_1", "min_sat_score_series",
    "max_sat_score_series", "grade_for_session_1"]

outcome_vars = ['grade_change', 'uptake_change', 'eliciting_change', 'talktime_change']

arms = ['control', 'tutor', 'tutor_student_social', 'tutor_student_personal']
outcomes = ['uptake', 'eliciting', 'grade', 'talktime']

ridge_weights = {
    'uptake': {
        # lambda = 
        'control': [-0.0278696318623852, 0.0189905583523548, 0.00390917807715487, -0.0112748997331886, 0.00548323001246631, 0.000383685852320607, 0.0059619016058133, -0.00242278747426645, 0.00911614529911483, 0.0102029880240925, -0.00360922000477894, 0.00755437148877794, -0.0218608852217323, -0.00595534908828803, 0.00727813586702457, -0.000641762637961628, 0.00767051622671632, -0.00399972033470474, -0.00399972033470474, 0.000740046858197032],
        # lambda = 
        'tutor': [-0.00448033538596491, 0.000881526557352097, 0.000402924338509245, 0.000598646846730125, 0.000610433973374744, -0.00103282581171821, 0.000376607510729873, -0.00178002676388304, 4.02980699724232e-05, 0.00133482856056054, -0.00106086462851953, 0.000498825788623304, -0.00114079045896625, 0.000891483508476126, 0.000357382907523636, 0.000477314688209969, 0.000760819668187678, -0.00101210171113324, -0.00101210171113324, -0.000983504855042292],
        # lambda = 
        'tutor_student_social': [-0.0194394829333333, 0.000177110896422636, -0.000326231144808998, -0.000348661877928616, -0.000162189891169474, 4.31312625715379e-05, -5.88374805912788e-06, -0.00026655400023335, -0.000663981803800357, -0.00107635234290931, 0.00125748876645729, 0.00032308382811703, -0.0010126828370249, -0.00116586724128377, 3.95079938375836e-05, -0.000755046525180819, -0.000402049974800908, -0.000845740989340836, -0.000845740989340836, -0.000773950563457796],
        # lambda = 
        'tutor_student_personal': [-0.0069741498275862, 0.000416883948053873, -0.00148705016780567, 0.000890502481745842, 0.000700835803270379, 0.00014272488240974, 0.000127573165518389, 0.000676882642945349, 7.85626715233299e-05, -0.000837268831269751, 0.0004057103374589, -0.000507070594495715, -0.000588109538539869, 0.000826438290717277, -0.000311056532711955, -3.68935648010603e-05, 0.000181862817692885, -0.000399404113632805, -0.000399404113632805, -0.000133303278523395]
    },
    'eliciting': {
        # lambda = 
        'control': [-0.0435910788073395, 0.00142465486491457, 0.00374680828754696, 0.000134652037552566, -0.00140312606229529, -0.000994208398563626, -0.00176487521024374, 0.00129753438101319, 0.000934976008553091, -0.000147199051292362, -0.00112099986740929, -0.000365373047379156, 0.000349278833972979, -0.00287864360956067, -0.000240846520397486, 0.00079279398977793, 0.00239487076515486, -0.000153466570228798, -0.000153466570228799, -0.000560675105530724],
        # lambda = 
        'tutor': [-0.00787257946491213, 0.0207988371632251, -0.00623155811321966, 0.0221082052414176, -0.0117201775711513, 0.0154082292118699, -0.0360921354435353, -0.014320677562699, -0.00493587727638165, -0.00927638695905871, 0.00868582393357103, -0.0016548920920019, -0.0196223217420385, -0.0474650910601967, -0.0101238911578985, 0.0155222089192333, -0.0167779031214161, 0.013166888712727, 0.013166888712727, -0.023621341102884],
        # lambda = 
        'tutor_student_social': [-0.0356637686333333, 3.77926303702373e-05, 0.00137442396701859, -0.000765267303872388, -0.000865266806252964, -0.000746645396693725, -0.000125751864958424, 0.00106300999562735, -0.000755358079316314, 0.0010414556030638, 4.06057620493888e-05, 0.000803077593123254, 0.00267533495096434, -0.000721353987670084, 0.000461132625394889, 3.63594131686244e-05, -0.000278589537687644, -0.000243632276248807, -0.000243632276248807, -0.000786891576313165],
        # lambda = 
        'tutor_student_personal': [-0.0391191465, -7.30390248294287e-05, 0.000845985240351025, 0.00179184027321836, 0.00112742904418674, 0.0013238178460483, 0.000357586744026654, -0.000639114434640772, -1.40705349058091e-05, -0.00152347020731424, 0.000582690520634938, -0.00014825816070494, -0.00131095207191597, -0.00271418626018386, -0.000420529734516442, -0.000771013014589733, -0.00117994311897714, 0.00131166634836541, 0.00131166634836541, 0.000525900610711853]
    },
    'grade': {
        # lambda = 
        'control': [0.0362241291651376, -0.00357562844838288, -0.00491752029445762, -0.00631338242037342, -0.00730731032824618, 0.00534997050299842, 0.00547018922178192, -0.000267781442331812, -0.00143702180011297, -0.001879315402642, -0.0109256640634882, -0.0106137249767205, 0.0110285733658138, 0.00150245776756513, 0.00146345055110332, -0.0119078338590153, -0.0077506559651179, 0.0103521015193069, 0.0103521015193068, -0.023606538363854],
        # lambda = 
        'tutor': [0.0206735338508772, -0.000259371830391793, -0.000528655262218599, 0.000522967129120938, 0.000577125922107853, -0.00235042004543068, 0.000866715012552675, -0.000291726163633884, -0.000257443543539203, 0.000876332944280698, -0.00050310975934437, 0.000651213264841046, -0.000232573898399287, 1.85433917938999e-05, 0.000153142023361557, 3.36903319580259e-05, 0.000488175183804603, 0.000390290138903258, 0.000390290138903257, -0.00129087936842214],
        # lambda = 
        'tutor_student_social': [0.0257364405833333, 0.00924895467617791, 0.00587064225635775, 0.00731799638154797, 0.00351245726341333, 0.0114955276593054, -0.0087109761079055, 0.00292112858884236, -0.00105344198764319, 0.0195077260986749, -0.00815543024084213, 0.00345492993366266, 0.000234742703745654, 0.00819359221646078, 0.00348655113951563, 0.00016528425158297, 0.000532707062702268, 0.0200519206127109, 0.0200519206127109, -0.0199147121812586],
        # lambda = 
        'tutor_student_personal': [0.0430595236551724, -0.000605332127204132, 0.000810299836785397, -5.1507397137908e-05, 0.000298404388027563, 9.34612803616786e-06, 7.52818032076676e-05, -0.000137845096740975, -0.00023747223540972, 0.000513665807285231, -0.000970178444332535, -6.26557950304339e-05, 0.00019888656262811, 0.000372871863380197, -0.000645759340522184, 0.000334691311003758, 0.000219097072191329, 2.30001949890019e-05, 2.30001949890019e-05, -0.000679572690850047]
    }, 
    'talktime': {
        # lambda = 
        'control': [-0.0285816079541285, -0.0031010045525868, 0.00881682924253457, 0.0119806320746078, 0.00665845844845414, 0.00199717053283706, -0.00124781379494559, -0.00396502577994172, -0.00288775452745431, -0.00204805997479151, 0.000724092838212748, -0.00296687470493707, 0.00301870322664853, 0.00349036796704154, -0.00446564031621344, 0.00290557970270182, -0.00352095169810337, 0.00430635064003513, 0.00430635064003512, 0.00333909739981458],
        # lambda = 
        'tutor': [-0.0164662387807018, 0.0030742194578544, 0.0104827359616458, 0.0452129554290429, -0.00507935135970459, 0.00947174937768481, -0.00494056737756901, -0.00203496605982262, -0.00505063004749925, 0.00184328412780669, 0.00695641566487729, -0.0170670694765909, 0.00632835025129885, -0.00481791271454458, 0.0177139755349796, -0.00124224514998695, 0.00744790178564515, 0.00765178034638762, 0.00765178034638762, -0.00894927167511633],
        # lambda = 
        'tutor_student_social': [-0.0256958377666663, 0.0130221244259397, -0.00223412080518029, 0.124868969777359, -0.00703545444241454, -0.0197003940857845, 0.00831955216004929, -0.0115150566463041, -0.0143589454279676, -0.00662416651348398, -0.00504669419454883, 0.0167373403542811, 0.0259058663168139, 0.0221327260759348, 0.031974647836046, -0.0214761619726569, 0.00567401867514714, -0.00882373189492915, -0.00882373189492934, 0.0185593362971043],
        # lambda = 
        'tutor_student_personal': [-0.0101180637931034, -0.00108575119047358, 0.000150580177809319, 0.00104413826705564, 0.000144483551572358, -0.000275577934853628, -5.20254536466232e-05, -0.000661250028144215, -0.000146171225551385, 0.000299712852933498, -5.65695976315138e-05, 3.59063205281697e-05, -0.00101093221801181, -0.000557297892060325, 8.1844553879847e-05, -4.42370479008088e-05, 1.50779305321868e-05, 0.000853965805744897, 0.000853965805744897, 0.000361403644587755]
    }, 
}

lasso_weights = {
    'uptake': {
        # lambda = 0.03238662
        'control': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.03238662
        'tutor': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.03238662
        'tutor_student_social': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.03238662
        'tutor_student_personal': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'eliciting': {
        # lambda = 0.02661865
        'control': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.02661865
        'tutor': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.02661865
        'tutor_student_social': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.02661865
        'tutor_student_personal': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'grade': {
        # lambda = 0.01335198
        'control': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.01335198
        'tutor': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.01335198
        'tutor_student_social': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.01335198
        'tutor_student_personal': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'talktime': {
        # lambda = 0.01773217
        'control': [0, 0, 0.0357091783761846, 0, 0, 0, -0.00779613217085455, -0.0201520900239028, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.01946103
        'tutor': [0, 0, 0.0335715756007803, 0, 0, 0, -0.00619365219179991, -0.0181470795266091, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.02344088
        'tutor_student_social': [0, 0, 0.0286508090831832, 0, 0, 0, -0.00250473958502252, -0.0135315407388191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # lambda = 0.02823462
        'tutor_student_personal': [0, 0, 0.022719941541301, 0, 0, 0, 0, -0.00780796675321372, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }, # TODO: add an aggreagated outcome measure??
}


    
summary_df = pd.read_csv('may_summary_data.csv')
samples_df = summary_df[covariates]
outcome_df = summary_df[outcome_vars]

# loop through each row in samples_df and get predicted policy for each outcome variable

# 1. CREATE SAMPLES MATRIX (num samples x num covariates)
samples = []
for index, row in samples_df.iterrows():
    ndarray = row.to_numpy()  # or row.values
    samples.append(ndarray)

samples_matrix = np.array(samples)

print(samples_matrix.shape)

# 2. CREATE WEIGHTS MATRIX FOR EACH OUTCOME VAR (num covariates x num arms) 20 x 4

weights = ridge_weights # <--- CHOOSE WHICH ONE YOU'RE DOING!!!

for reward in weights:
    #print(weights[reward])
    for treatment in weights[reward]:
        #print(weights[reward][treatment])
        #print(len(weights[reward][treatment][0:10]))
        #print(len(weights[reward][treatment][11:]))
        temp = weights[reward][treatment][0:10]
        temp.extend(weights[reward][treatment][11:])
        #weights[reward][treatment][0:10].extend(weights[reward][treatment][11:])
        weights[reward][treatment] = temp

# uptake
curr_dict = weights['uptake']
#print(len(curr_dict['control']))
#0/0
uptake_weights = np.array([curr_dict['control'], curr_dict['tutor'], curr_dict['tutor_student_social'], curr_dict['tutor_student_personal']]).T

# eliciting
curr_dict = weights['eliciting']
eliciting_weights = np.array([curr_dict['control'], curr_dict['tutor'], curr_dict['tutor_student_social'], curr_dict['tutor_student_personal']]).T

# grade
curr_dict = weights['grade']
grade_weights = np.array([curr_dict['control'], curr_dict['tutor'], curr_dict['tutor_student_social'], curr_dict['tutor_student_personal']]).T

# talktime
curr_dict = weights['talktime']
talktime_weights = np.array([curr_dict['control'], curr_dict['tutor'], curr_dict['tutor_student_social'], curr_dict['tutor_student_personal']]).T
                        

# 3. GET SCORES VAR (num samples x num arms) 
# multiply sample matrix by weights matrix for each outcome var

uptake_scores = np.dot(samples_matrix, uptake_weights)
eliciting_scores = np.dot(samples_matrix, eliciting_weights)
grade_scores = np.dot(samples_matrix, grade_weights)
talktime_scores = np.dot(samples_matrix, talktime_weights)

# 4. GET PREDICTED POLICY (num samples x 1)
# take argmax of scores matrix

uptake_policy = np.argmax(uptake_scores, axis=1)
eliciting_policy = np.argmax(eliciting_scores, axis=1)
grade_policy = np.argmax(grade_scores, axis=1)
talktime_policy = np.argmax(talktime_scores, axis=1)

# 5. ADD PREDICTED POLICY AS COLUMNS IN SUMMARY_DF

summary_df['uptake_policy'] = uptake_policy
summary_df['eliciting_policy'] = eliciting_policy
summary_df['grade_policy'] = grade_policy
summary_df['talktime_policy'] = talktime_policy

mapping = {0: 'Control', 1: 'Tutor Feedback', 2: 'Tutor Feedback + Goal-Oriented Learner Feedback', 3: 'Tutor Feedback + Socially-Oriented Learner Feedback'}

# Custom function to replace each value based on the mapping
def replace_argmax(value):
    return mapping.get(value, 'Unknown')  # Use 'Unknown' for any value not in the mapping

# Apply the custom function to the 'name' column
summary_df['uptake_policy'] = summary_df['uptake_policy'].apply(replace_argmax)
summary_df['eliciting_policy'] = summary_df['eliciting_policy'].apply(replace_argmax)
summary_df['grade_policy'] = summary_df['grade_policy'].apply(replace_argmax)
summary_df['talktime_policy'] = summary_df['talktime_policy'].apply(replace_argmax)

# 6. EXPORT SUMMARY_DF TO CSV (just so we have it)

summary_df.to_csv('summary_data_with_policy_ridge_may_new.csv')