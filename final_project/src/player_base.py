from game.players import BasePokerPlayer
import random as rand


class BasePlayer(BasePokerPlayer):
    def __init__(self):
        self.fold_ratio = self.call_ratio = raise_ratio = 1.0 / 3
        # 12345 is high card
        # 0: high, 1:one pair, 2:two pair, 3:三條,4: 順, 5:同花, 
        # 6: 葫蘆 , 7:鐵支, 8:同花順
        self.type = [0,1,2,3,4,5,6,7,8]
        self.playernum = -1

    # 找到自己為Player幾
    def decide_playernum(self, round_state):
        seat = round_state['seats']
        for i in range(0 , len(seat)):
            if seat[i]['uuid'] == self.uuid:
                return i+1
        return -1   

    # 0: high, 1:one pair, 2:two pair, 3:三條,4: 順, 5:同花, 
    # 6: 葫蘆 , 7:鐵支, 8:同花順
    # 翻前手牌類型
    def decide_pretype(self, hole_card):
        now_type = 0
        max_num = 0 # 幾對，幾high
        min_num = 0
        same = 0 # 同花不同花
        first = hole_card[0]
        second = hole_card[1]
        if (first[0] == second[0]):
            same = 1
        if (first[1] == second[1]):
            now_type = 1
        dic_now = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13}

        first_val = dic_now[str(first[1])]
        second_val = dic_now.get(str(second[1]))
        max_num = max(first_val, second_val)
        min_num = min(first_val, second_val)
        return now_type, max_num, min_num, same

    # 決定是否全下和全棄牌 (翻前)
    def decide_extreme(self, round_state):
        # 0:全下 1:全棄 2:正常
        final_decision = 2

        if ((round_state['seats'][self.playernum-1]['stack'] - round_state['seats'][2-self.playernum]['stack']) \
            > (20-self.round) * 2*round_state['small_blind_amount']):
            final_decision = 1
        elif (abs(round_state['seats'][self.playernum-1]['stack'] - round_state['seats'][2-self.playernum]['stack']) \
            > ((20-self.round) * 2*round_state['small_blind_amount'] + round_state['small_blind_amount'] * 2)):
            final_decision = 0
        return final_decision

    def all_in(self, valid_actions):
        num = (valid_actions[2]['amount']['max'])
        num = max(num, 0)
        return num

    # valid: 0:fold, 1:call, 2:raise
    # street: preflop flop turn river
    def declare_action(self, valid_actions, hole_card, round_state):
        self.playernum = self.decide_playernum(round_state)
        action = valid_actions[1]['action']
        amount = valid_actions[1]['amount']
        #print(hole_card)

        if (round_state['street'] == 'preflop'):
            pre_type, pre_maxnum, pre_minnum, pre_same = self.decide_pretype(hole_card)
            
            if ((pre_type == 1 and pre_maxnum >= 10) or (pre_minnum == 1) or \
                (pre_same == 1 and pre_minnum >= 10)):

                action = valid_actions[2]['action']
                amount = self.all_in(valid_actions)
                return action, amount
            
            action = valid_actions[1]['action']
            amount = valid_actions[1]['amount']
            return action, amount

        action = valid_actions[1]['action']
        amount = valid_actions[1]['amount']

        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return BasePlayer()
