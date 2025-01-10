import json, os
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from src.player_base import setup_ai as player_base_ai
from agents.console_player import setup_ai as console_ai
from src.monto_player import setup_ai as monto_ai
from src.final_player import setup_ai as final_ai
from src.RL_DQM import setup_ai as RL_DQM_ai
from src.agent import setup_ai as agent_ai


from baseline0 import setup_ai as baseline0_ai # type: ignore
from baseline1 import setup_ai as baseline1_ai # type: ignore
from baseline2 import setup_ai as baseline2_ai # type: ignore
from baseline3 import setup_ai as baseline3_ai # type: ignore
from baseline4 import setup_ai as baseline4_ai # type: ignore
from baseline5 import setup_ai as baseline5_ai # type: ignore
from baseline6 import setup_ai as baseline6_ai # type: ignore
from baseline7 import setup_ai as baseline7_ai # type: ignore

# initial
# config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)

config.register_player(name="p1", algorithm=baseline0_ai())
config.register_player(name="p2", algorithm=agent_ai())

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
game_result = start_poker(config, verbose=1)
#print(game_result)
#print(json.dumps(game_result, indent=4))
print(game_result['players'][1]['stack'])