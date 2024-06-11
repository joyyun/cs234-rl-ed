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
        'control': [-0.0294864288501205, 0.0141857054100796, 0.00445412084492997, -0.0133736607877092, 0.00381084317455474, 0.00925728947758921, -0.000531905327974887, 0.00366040686850766, 0.00642756419479235, 0.00535135263008866, -0.00133816711688814, -0.00266177745269875, -0.025830842643614, 0.00778018776771444, 0.00197527058438039, 0.00272532894932364, 0.0111229007456334, -0.00649779547796338, -0.00649779547796338, 0.00580455151350367],
        # lambda = 
        'tutor': [-0.0185515043721692, 0.0071722484620885, -0.00707422409784267, -0.00256187467112137, -0.00304389393844064, -0.00177106384185048, 0.0117656526691292, -0.0020733483641864, 0.00183485731100005, 0.00915418194533051, -0.00258873527175893, 0.0025564992864571, -0.0193874730061251, 0.00976718273766075, 0.00756142685195416, 0.00401398523000233, 0.00479477581643791, -0.000956228875345377, -0.000956228875345388, -0.00281774365857235],
        # lambda = 
        'tutor_student_social': [-0.0181911781583211, 0.00146142138231659, -0.00196582485636841, -0.000370220802495573, 0.000149800974499215, -0.000609283854199375, 0.000213470438941318, 0.00189798036452608, 0.000418054588179244, -0.000918722049105287, -0.000377799081010618, -0.000114074249052614, -0.00183362152196717, 0.000143556506828816, 0.000189331697587262, -0.000742167715080306, -0.000297157745846413, 0.000713547663722525, 0.000713547663722525, -0.000303556898622287],
        # lambda = 
        'tutor_student_personal': [-0.0219727212524332, 0.00385391210520978, 0.0106584062599687, -0.007570925383134, 0.00467603276704972, 0.00817068392340624, 0.00530716908254593, -0.00179501008484005, -0.00418194431355744, -0.00780496719130895, 0.0117458424140541, 0.0036334398427951, -0.00881533511752079, -0.00170310499511733, 0.00418612044909704, 0.00117254476562878, 0.00212803202330567, 0.000625510588575733, 0.000625510588575736, -0.00846555365070019],
    },
    'eliciting': {
        # lambda = 
        'control': [-0.0432819504445296, -0.00567652212884413, 0.0119812144754175, 0.0058289683825414, -0.00447431160678268, 0.00163259890661724, -0.0117612130516728, 0.0179457254331547, 0.00589163111508347, -0.00795860857085409, -0.00211542977270396, 0.0088029502957344, -0.00804019625430354, -0.0307737882157378, 0.00072619971900646, 0.00669180139137244, 0.00107039423017293, -0.0100513409167339, -0.0100513409167339, -0.010134096171027],
        # lambda = 
        'tutor': [-0.0299745172688232, 0.0025181753661232, -0.00312369967602513, 0.0113789316174334, -0.00685264506425226, -0.00181650879401272, -0.0116749664060141, 0.00218478036292453, 0.00642143847545917, -0.0039034107622432, 0.00111955629323827, 0.0041160540308323, -0.00807226439900221, -0.0558759412300746, -0.00908876126248617, 0.00816476937319887, -0.012491483089785, 0.00648271413981013, 0.00648271413981009, -0.0120640635684663],
        # lambda = 
        'tutor_student_social': [-0.022834181272049, 0.00109228965128155, 0.00141582911737799, 0.00193269088818333, 0.000547925386355376, -0.000962197526695072, 0.00081323200815099, 0.00110757364628965, -0.000668295462827605, -0.00110873562398519, -0.000123700494274374, -0.00072171978613522, -0.000119057461582559, -0.00277368302701239, -0.0013519601708567, -0.00068546752981147, -0.000630608681850054, -0.000270867555913014, -0.000270867555913016, 0.000274396548230635],
        # lambda = 
        'tutor_student_personal': [-0.0583165358155155, 0.00193497807002732, 4.43368557916347e-05, 0.000990693054772009, 0.000126652972742705, -0.000515868394064122, 0.000241848266916366, 0.000669533051974743, -0.000503842305739143, 0.00409292656463139, 0.000445172001684055, -0.001642115442074, -0.00123918910412688, -0.00579237706097584, -0.00250967128907678, -0.00269314107378083, -0.00656182294359027, -0.00124625640263815, -0.00124625640263815, 0.00120792983079942],
    },
    'grade': {
        # lambda = 
        'control': [0.0786116780911678, -0.00683779220926938, -0.012438539389175, -0.00358043321831898, 0.00649826934050066, -0.00591620009389047, 0.00488748328357329, 0.00469159397829243, 0.00536817431936166, -0.0166679030478011, 0.011534043890821, -0.0271798925680291, -0.0107305958965048, 0.00821907266207935, 0.0301562164990907, -0.00145811195094449, 0.00741571936094232, 0.0149147535590249, 0.0149147535590249, -0.0308943066182499],
        # lambda = 
        'tutor': [0.0661425930771169, -0.00883270997838469, 0.00113174370015938, -0.00279851384380031, 0.000420329756264752, 0.00215094438365552, 0.00140167998513767, 0.00113623243013778, 0.00225833347590671, 0.0034440553076797, -0.0026961092879432, -0.00267695065135584, 0.00253895502816096, 0.000955231240798779, 0.00289998048859413, 0.00165296999688256, 0.0020682960958846, 0.00212153647862721, 0.0021215364786272, -0.00369450685581242],
        # lambda = 
        'tutor_student_social': [0.0697725328345963, -0.00311669766473958, -0.000770934821206492, -0.000655807181691914, -0.00155373443244091, 0.000351226610340808, -0.000318659446746463, 0.00154101274890626, -0.000681270625672567, 0.000748408172967025, -0.00130636645786243, -7.56913592973347e-05, 0.00100838356266241, -0.000401766194742413, 0.00247183326916118, 0.000672080878441022, 0.000942610756236856, 0.00126622247281116, 0.00126622247281116, -0.000206474894100866],
        # lambda = 
        'tutor_student_personal': [0.078600324295837, -0.0121869062496264, -0.00801896031963014, 0.00275443090081201, 0.0112991886605728, 0.00709088733566374, -0.00828971787085769, 0.00753496929543303, 0.00475889969899202, 0.00234516009087413, -0.00539400682023334, -0.0174491242142374, 0.00599940080960281, 0.0030200509692494, 0.0340718454300749, 0.0159658239601079, -0.00176815452420859, 0.0178409577488208, 0.0178409577488209, -0.0299004221389679],
    }, 
    'talktime': {
        # lambda = 
        'control': [-0.0213983995579452, -0.00432292062758106, 0.00186129260845821, 0.0187814122376992, 0.00576279774791715, 0.00281339572187086, 0.000479702831379055, -0.00379939795028999, 0.00139066281869026, 0.000599986469843706, 0.00289367448557689, -0.00367025614156767, 0.00027864617359792, 0.00645953347802636, 0.0047713821274971, 0.00708779814524717, -0.00353920541839093, 0.00660805018556954, 0.00660805018556955, 0.00374032182555888],
        # lambda = 
        'tutor': [-0.0177998844741972, -0.0108840813848982, 0.0101652370377692, 0.0775284935601696, -0.0134041771289552, 0.00661872806279065, -0.00292252269881851, -0.00157409309618056, -0.00531217541282431, 0.00066339282562993, 0.00694610445353333, 0.00307617979722668, 0.00860289861712742, -0.00580779691546534, 0.00475290451229728, -0.00507479095564265, 0.0111898535145088, 0.00251666322127574, 0.00251666322127565, 0.0092408503347564],
        # lambda = 
        'tutor_student_social': [-0.0234893953589861, -0.00746869163750272, -0.00596901299841161, 0.0766603716821685, -0.000779802323225226, -0.0163610191723378, -0.017216195621611, 0.00185353477219785, -0.00395267963460912, -0.00550196077960453, 0.0064550329054404, -0.0269692704932756, -0.024178433486178, -0.00518288129994655, 0.00496485940017788, 0.0498084687116614, 0.0034958045652586, -0.00241912671520671, -0.00241912671520691, 0.0124198681988322],
        # lambda = 
        'tutor_student_personal': [-0.0352637888418595, -0.00333099540686286, 0.00236925302612274, 0.0837088629915852, -0.00102281934900719, -0.0076899920483565, 0.000475509836623345, -0.00714650276570121, -0.0027717648910628, 0.00439642454790023, 0.00121077489398763, 0.00579081965131519, 0.0092245844580168, 0.00486694916466281, -0.00186926486500018, 0.0242629395689407, -0.00714525326090928, 0.00875963038930546, 0.00875963038930548, -0.000855573645592763],
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


    
summary_df = pd.read_csv('full_data_splits/agg_test_data.csv')
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

summary_df.to_csv('agg_data_with_policy_ridge.csv')