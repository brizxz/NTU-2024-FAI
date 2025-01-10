import sys
sys.path.insert(0, '../')
from game.players import BasePokerPlayer
from src.tools import *

class FinalPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.round = 0
        self.n_simulations = 10000
        self.playernum = -1

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
        # Estimate the win rate
        if (round_state['street'] == 'preflop'):
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

        win_rate = estimate_hole_card_win_rate(self.n_simulations, len(round_state['seats']), hole_card, round_state['community_card'])
        # Check whether it is possible to call
        can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
        if can_call:
            # If so, compute the amount that needs to be called
            call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
        else:
            call_amount = 0

        amount = None

        # If the win rate is large enough, then raise
        if win_rate > 0.5:
            raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']
            if win_rate > 0.8:
                # If it is extremely likely to win, then raise as much as possible
                action = 'raise'
                if (round_state['street'] == 'preflop'):
                    amount = 100
                else:
                    amount = raise_amount_options['max']
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

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        #print(hole_card)
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # print(hand_info)
        pass
        
def setup_ai():
    return FinalPlayer()