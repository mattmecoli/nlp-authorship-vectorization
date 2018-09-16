# Evaluating black box models is always challenge and requires a bit of creativity
# We have chosen to use two methodologies: 1) scaled scoring and 2) analoglical reasoning TK - add in "as first used in Markloc yadda yadda"


#the similarties come out as a list of tuples
# sims[x][y]
# x is 0-22 (one for each tag, in order)
# y is 0 or 1 where 0 is the tag and 1 is the cosine similarity


# will need a length for how many loops to do
# will need to pass the model through
# we may want to measure which of them is doing best
# male/female, period, or author

def get_model_metrics(model,   ,random=False, rand_samp_size = 500):

    # if random=True:
    #     #TK - Do something here that gets a random number of samples
    #
    # else:
    #     continue #TK

    for x in model:
        point_value = len(!!!!) #TK
        count = 0
        correct_labels = []

        scores_for_avg_metric = []
        scores_for_wght_metric = []

        while count <= (len(!!!!)-1): #TK


            if sims[count][0] is in correct_labels:
                scores_for_avg_metric.append(sims[count][1])
                scores_for_wght_metric.append(((score_for_average_metric * point_value) / len(!!!!))) #NOT SURE ABOUT THIS TK

            else:
                continue
            point_value -= 1
            count +=1

        #PERSIST THE SCORES AND TIE THEM TO EACH INDIVIDUAL DOC TK
        #E
