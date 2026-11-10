from .base_agent import BaseAgent, Observation, ACTIONS, ACTION_NAMES, ROLES, own_utility, best_response_policy
from .deterministic_agent import DeterministicAgent
from .frequency_agent import FrequencyAgent
from .fictitious_play_agent import FictitiousPlayAgent
from .minimax_agent import MinimaxAgent, solve_maximin

AGENTS = {"deterministe": DeterministicAgent, "frequences": FrequencyAgent,
          "fictitious_play": FictitiousPlayAgent, "minimax": MinimaxAgent}
