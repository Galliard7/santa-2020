import os
import json
import pandas as pd

# Data folder
data_loc = '../input/santa-2020-top-agents-dataset/episode/'

# Collect json file names to parse
files = []
for file in os.listdir(data_loc):
    files.append(data_loc+file)


################
# Parse data
################

episodeData = pd.DataFrame()

for num, file in enumerate(files):

    # Progress
    if num % 50 == 0:
        print(num)
    # Open file
    f = open(file)

    # Load json data
    data = json.load(f)

    # Parse Game id
    start = file.rfind('/')
    end = file.find('.json')
    gameid = file[start+1:end]

    # Loop through each step
    for step, stepData in enumerate(data['steps']):

        # Progress
        # if step % 500 == 0:
        #     print(step)

        # New row 1
        row1 = {}
        # Game id
        row1['gameID'] = gameid
        # Agent index
        row1['agentIdx'] = stepData[0]['observation']['agentIndex']
        # Step number
        row1['step'] = step
        # Action 1
        row1['self_action'] = stepData[0]['action']
        # Action 2
        row1['opp_action'] = stepData[1]['action']
        # Reward
        row1['reward_total'] = stepData[0]['reward']
        # Append row
        episodeData = episodeData.append(row1, ignore_index=True)


        # New row 2
        row2 = {}
        # Game id
        row2['gameID'] = gameid
        # Agent index
        row2['agentIdx'] = stepData[1]['observation']['agentIndex']
        # Step number
        row2['step'] = step
        # Action 1
        row2['self_action'] = stepData[1]['action']
        # Action 2
        row2['opp_action'] = stepData[0]['action']
        # Reward
        row2['reward_total'] = stepData[1]['reward']
        # Append row
        episodeData = episodeData.append(row2, ignore_index=True)



    # Close file
    f.close()

    print("{} parsed".format(file))

############################
# Calculate Step Rewards
############################

# Sort dataframe
episodeData.sort_values(by=['gameID','agentIdx', 'step'], inplace=True)

# Reset indexes
episodeData.reset_index(drop=True, inplace=True)

# Current game
curGame = -1
curAgent = -1

reward = 0

stepReward = []

print(episodeData.shape)

for row in episodeData.itertuples():

    # Index
    ind = getattr(row,'Index')

    # Progress
    if ind % (episodeData.shape[0]//10) == 0:
        print(ind)

    if getattr(row,'gameID') != curGame:
        curGame = getattr(row,'gameID')
        curAgent = getattr(row,'agentIdx')

    if getattr(row,'agentIdx') != curAgent:
        curAgent = getattr(row,'agentIdx')

    try:
        if ind == 0:
            reward = 0
        elif episodeData.loc[ind-1,'gameID'] == curGame and episodeData.loc[ind-1,'agentIdx'] == curAgent:
            reward = getattr(row,'reward_total') - episodeData.loc[ind-1,'reward_total']
        else:
            reward = 0

        stepReward.append(reward)

    except Exception as e:
        print(e)
        print(getattr(row, 'Index'), getattr(row, 'gameID'), getattr(row, 'agentIdx'), getattr(row, 'step'))
        break




episodeData['reward_step'] = stepReward
print(episodeData.shape)

episodeData.to_csv("../modelData/SantaEpisodeData.csv", index=False)
