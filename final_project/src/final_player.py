import torch # type: ignore
import torch.nn as nn # type: ignore

import numpy as np

import sys
sys.path.insert(0, '../')
from game.players import BasePokerPlayer

from src.tools import *

import pickle

class DQNAgent(nn.Module):
    def __init__(
        self,
        h_size=128,
        total_num_actions=5,
        is_double=False,
        is_main=True,
        is_train=True
    ):
        super(DQNAgent, self).__init__()  
        self.h_size = h_size
        self.total_num_actions = total_num_actions
        self.is_double = is_double
        self.is_main = is_main
        self.is_train = is_train
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, h_size, kernel_size=5),
            nn.ReLU(),
            
        )
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(h_size + 128, 256)
        self.fc4 = nn.Linear(256, h_size)
        if is_double:
            self.AW = nn.Parameter(torch.Tensor(self.h_size // 2, total_num_actions))
            self.VW = nn.Parameter(torch.Tensor(self.h_size // 2, 1))
            torch.nn.init.xavier_uniform_(self.AW)
            torch.nn.init.xavier_uniform_(self.VW)
        else:
            self.fc5 = nn.Linear(h_size, total_num_actions)
    def forward(self, scalar_input, features_input):
        scalar_input = scalar_input.view(-1, 1, 17, 17)
        conv_out = self.conv(scalar_input).view(-1, self.h_size)
        fc1_out = self.fc1(features_input)
        fc2_out = self.fc2(fc1_out)
        
        merged = torch.cat((conv_out, fc2_out), dim=1)
        fc3_out = self.fc3(merged)
        fc4_out = self.fc4(fc3_out)
        
        if self.is_double:
            self.stream_A, self.stream_V = torch.split(fc4_out, self.h_size // 2, dim=1)
            advantage = torch.matmul(self.stream_A, self.AW)
            value = torch.matmul(self.stream_V, self.VW)

            q_out = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_out = self.fc5(fc4_out)
        
        return q_out

class DQNPlayer(BasePokerPlayer):
    def __init__(
        self,
        h_size = 128,
        total_num_actions = 5,
        is_double = False,
        is_main = True,
        is_restore = False,
        is_train = True,
        round = 0,
        playnum = -1
    ):
        
        
        self.h_size = h_size
        self.total_num_actions = total_num_actions
        self.is_double = is_double
        self.is_main = is_main
        self.is_restore = is_restore
        self.is_train = is_train
        self.round = round
        self.playernum = playnum
        
        # Hole card winning rate
        with open('./card_estimation.pkl', 'rb') as f:
            self.hole_card_est = pickle.load(f)
        # with open('./src/hole_card_estimation.pkl', 'rb') as f:
        #     self.hole_card_est = pickle.load(f)

        self.model = DQNAgent(
            h_size,
            total_num_actions,
            is_double,
            is_main,
            is_train)
        
        if is_restore:
            # print("Load model...")
            self.model.load_state_dict(torch.load('./model_base.pth'))
            #self.model.load_state_dict(torch.load('./src/model_MC_2.pth'))

        
    
    def decide_playernum(self, round_state):
        seat = round_state['seats']
        for i in range(0 , len(seat)):
            if seat[i]['uuid'] == self.uuid:
                return i+1
        return -1   

    def decide_extreme(self, round_state):
        # 0:全下 1:全棄 2:正常
        final_decision = 2
        if ((round_state['seats'][self.playernum-1]['stack'] - round_state['seats'][2-self.playernum]['stack']) \
            > (20-self.round) * 2*round_state['small_blind_amount']):
            # print(round_state['seats'][self.playernum-1]['stack'])
            final_decision = 1
        elif (abs(round_state['seats'][self.playernum-1]['stack'] - round_state['seats'][2-self.playernum]['stack']) \
            > ((20-self.round) * 2*round_state['small_blind_amount'] + round_state['small_blind_amount'] * 2)):
            final_decision = 0
            
        return final_decision

    def all_in(self, valid_actions):
        num = (valid_actions[2]['amount']['max'])
        num = max(num, 0)
        return num

    def declare_action(self, valid_actions, hole_card, round_state):
        win_rate = estimate_hole_card_win_rate(4000, len(round_state['seats']), hole_card, round_state['community_card'])
        if (round_state['street'] == 'preflop'):
            #print(hole_card)
            self.round += 1
            self.playernum = self.decide_playernum(round_state)
            final_decision1 = self.decide_extreme(round_state)
            if final_decision1 == 0:
                action = valid_actions[2]['action']
                amount = self.all_in(valid_actions)
                return action, amount
            elif final_decision1 == 1:
                action = valid_actions[0]['action']
                amount = valid_actions[0]['amount']
                return action, amount
            
            # Check whether it is possible to call
            can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
            if can_call:
                # If so, compute the amount that needs to be called
                call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
            else:
                call_amount = 0

            amount = None

            if win_rate > 0.4:
                raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']
                if win_rate > 0.8:
                    # If it is extremely likely to win, then raise as much as possible
                    amount = 50
                elif win_rate > 0.7:
                    # If it is likely to win, then raise by the minimum amount possible
                    action = 'raise'
                    amount = raise_amount_options['min']
                else:
                    # If there is a chance to win, then call
                    action = 'call'
            else:
                action = 'call' if can_call and call_amount == 0 else 'fold'

            # Set the amount
            if amount is None:
                items = [item for item in valid_actions if item['action'] == action]
                amount = items[0]['amount']

            if (amount == -1):
                action = 'call'
                amount = 0
            return action, amount
        else:
            street = round_state['street']
            bank = round_state['pot']['main']['amount']
            stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
            other_stacks = [s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid]
            dealer_btn = round_state['dealer_btn']
            small_blind_pos = round_state['small_blind_pos']
            big_blind_pos = round_state['big_blind_pos']
            next_player = round_state['next_player']
            round_count = round_state['round_count']
            estimation = self.hole_card_est[(hole_card[0], hole_card[1])]
            
            features = get_street(street)
            features.extend([bank, stack, dealer_btn, small_blind_pos, big_blind_pos, next_player, round_count])
            features.extend(other_stacks)
            features.append(estimation)
            
            img_state = img_from_state(hole_card, round_state)
            img_state = process_img(img_state)
            img_state_tensor = torch.tensor(img_state, dtype=torch.float32)
            features_tensor = torch.tensor(features, dtype=torch.float32)
            action_num = torch.argmax(self.model(img_state_tensor.unsqueeze(0), features_tensor.unsqueeze(0)), axis=1)[0]
            action, amount = get_action_by_num(action_num, valid_actions)    

            # Check whether it is possible to call
            # can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
            # if can_call:
            #     # If so, compute the amount that needs to be called
            #     call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
            # else:
            #     call_amount = 0

            if win_rate > 0.85:
                # If it is extremely likely to win, then raise as much as possible
                action = 'raise'
                amount = 100
            elif win_rate > 0.75:
                # If it is likely to win, then raise by the minimum amount possible
                action = 'raise'
                amount = raise_amount_options['min']


        if (amount == -1):
            action = 'call'
            amount = 0

        return action, amount
    
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
            
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
        
        
def setup_ai(h_size=128, is_main=True, is_double=True, is_restore=True):
    return DQNPlayer(h_size=128, is_main=is_main, is_double=is_double, is_restore=is_restore)