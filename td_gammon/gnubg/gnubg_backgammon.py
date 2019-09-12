import os
import sys
import time
from collections import namedtuple
from itertools import count
import requests
from gym_backgammon.envs.backgammon import Backgammon as Game, WHITE, BLACK, NUM_POINTS, COLORS, assert_board
from gym_backgammon.envs.backgammon_env import STATE_W, STATE_H, SCREEN_W, SCREEN_H
from gym_backgammon.envs.rendering import Viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

gnubgState = namedtuple('GNUState', ['agent', 'roll', 'move', 'board', 'double', 'winner', 'n_moves', 'action', 'resigned', 'history'])


class GnubgInterface:
    def __init__(self, host, port):
        self.url = "http://{}:{}".format(host, port)
        # Mapping from gnu board representation to representation used by the environment
        self.gnu_to_idx = {23 - k: k for k in range(NUM_POINTS)}
        # In GNU Backgammon, position 25 (here 24 because I start from 0-24) represents the 'bar' move
        self.gnu_to_idx[24] = 'bar'
        self.gnu_to_idx[-1] = -1

    def send_command(self, command):
        try:
            resp = requests.post(url=self.url, data={"command": command})
            return self.parse_response(resp.json())
        except Exception as e:
            print("Error during connection to {}: {} (Remember to run gnubg -t -p bridge.py)".format(self.url, e))

    def parse_response(self, response):
        gnubg_board = response["board"]
        action = response["last_move"][-1] if response["last_move"] else None

        info = response["info"][-1] if response["info"] else None

        winner = None
        n_moves = 0
        resigned = False
        double = False
        move = ()
        roll = ()
        agent = None

        if info:
            winner = info['winner']
            n_moves = info['n_moves']
            resigned = info['resigned']

        if action:

            agent = WHITE if action['player'] == 'O' else BLACK

            if action['action'] == "double":
                double = True
            elif 'dice' in action:
                roll = tuple(action['dice'])
                roll = (-roll[0], -roll[1]) if agent == WHITE else (roll[0], roll[1])

            if action['action'] == 'move':
                move = tuple(tuple([self.gnu_to_idx[a - 1], self.gnu_to_idx[b - 1]]) for (a, b) in action['move'])

        return gnubgState(agent=agent, roll=roll, move=move, board=gnubg_board[:], double=double, winner=winner, n_moves=n_moves, action=action, resigned=resigned, history=response["info"])

    def parse_action(self, action):
        result = ""
        if action:
            for move in action:
                src, target = move
                if src == 'bar':
                    result += "bar/{},".format(target + 1)
                elif target == -1:
                    result += "{}/off,".format(src + 1)
                else:
                    result += "{}/{},".format(src + 1, target + 1)

        return result[:-1]  # remove the last semicolon


class GnubgEnv:
    DIFFICULTIES = ['beginner', 'intermediate', 'advanced', 'world_class']

    def __init__(self, gnubg_interface, difficulty='beginner', model_type='nn'):
        self.game = Game()
        self.current_agent = WHITE
        self.gnubg_interface = gnubg_interface
        self.gnubg = None
        self.difficulty = difficulty
        self.is_difficulty_set = False
        self.model_type = model_type
        self.viewer = None

    def step(self, action):
        reward = 0
        done = False

        if action and self.gnubg.winner is None:
            action = self.gnubg_interface.parse_action(action)
            self.gnubg = self.gnubg_interface.send_command(action)

        if self.gnubg.double and self.gnubg.winner is None:
            self.gnubg = self.gnubg_interface.send_command("take")

        if self.gnubg.agent == WHITE and self.gnubg.action['action'] == 'move' and self.gnubg.winner is None:
            if self.gnubg.winner != 'O':
                self.gnubg = self.gnubg_interface.send_command("accept")
                assert self.gnubg.winner == 'O', print(self.gnubg)
                assert self.gnubg.action['action'] == 'resign' and self.gnubg.agent == 1 and self.gnubg.action['player'] == 'X'
                assert self.gnubg.resigned

        self.update_game_board(self.gnubg.board)

        observation = self.game.get_board_features(self.current_agent) if self.model_type == 'nn' else self.render(mode='state_pixels')

        winner = self.gnubg.winner
        if winner is not None:
            winner = WHITE if winner == 'O' else BLACK

            if winner == WHITE:
                reward = 1
            done = True

        return observation, reward, done, winner

    def reset(self):
        # Start a new session in gnubg simulator
        self.gnubg = self.gnubg_interface.send_command("new session")

        if not self.is_difficulty_set:
            self.set_difficulty()

        roll = None if self.gnubg.agent == BLACK else self.gnubg.roll

        self.current_agent = WHITE
        self.game = Game()
        self.update_game_board(self.gnubg.board)

        observation = self.game.get_board_features(self.current_agent) if self.model_type == 'nn' else self.render(mode='state_pixels')
        return observation, roll

    def update_game_board(self, gnu_board):
        # Update the internal board representation with the representation of the gnubg program
        # The gnubg board is represented with two list of 25 elements each, one for each player
        gnu_positions_black = gnu_board[0]
        gnu_positions_white = gnu_board[1]
        board = [(0, None)] * NUM_POINTS

        for src, checkers in enumerate(gnu_positions_white[:-1]):
            if checkers > 0:
                board[src] = (checkers, WHITE)

        for src, checkers in enumerate(reversed(gnu_positions_black[:-1])):
            if checkers > 0:
                board[src] = (checkers, BLACK)

        self.game.board = board
        # the last element represent the checkers on the bar
        self.game.bar = [gnu_positions_white[-1], gnu_positions_black[-1]]
        # update the players position
        self.game.players_positions = self.game.get_players_positions()
        # off bar
        self.game.off = [15 - sum(gnu_positions_white), 15 - sum(gnu_positions_black)]
        # Just for debugging
        # self.render()
        assert_board(None, self.game.board, self.game.bar, self.game.off)

    def get_valid_actions(self, roll):
        return self.game.get_valid_plays(self.current_agent, roll)

    def set_difficulty(self):
        self.is_difficulty_set = True

        self.gnubg_interface.send_command('set automatic roll off')
        self.gnubg_interface.send_command('set automatic game off')

        if self.difficulty == 'beginner':
            self.gnubg_interface.send_command('set player gnubg chequer evaluation plies 0')
            self.gnubg_interface.send_command('set player gnubg chequer evaluation prune off')
            self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.060')
            self.gnubg_interface.send_command('set player gnubg cube evaluation plies 0')
            self.gnubg_interface.send_command('set player gnubg cube evaluation prune off')
            self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.060')

        elif self.difficulty == 'intermediate':
            self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.040')
            self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.040')

        elif self.difficulty == 'advanced':
            self.gnubg_interface.send_command('set player gnubg chequer evaluation plies 0')
            self.gnubg_interface.send_command('set player gnubg chequer evaluation prune off')
            self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.015')
            self.gnubg_interface.send_command('set player gnubg cube evaluation plies 0')
            self.gnubg_interface.send_command('set player gnubg cube evaluation prune off')
            self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.015')

        elif self.difficulty == 'world_class':
            self.gnubg_interface.send_command('set player gnubg chequer evaluation plies 2')
            self.gnubg_interface.send_command('set player gnubg chequer evaluation prune on')
            self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.000')
            
            self.gnubg_interface.send_command('set player gnubg movefilter 1 0 0 8 0.160')
            self.gnubg_interface.send_command('set player gnubg movefilter 2 0 0 8 0.160')
            self.gnubg_interface.send_command('set player gnubg movefilter 3 0 0 8 0.160')
            self.gnubg_interface.send_command('set player gnubg movefilter 3 2 0 2 0.040')
            self.gnubg_interface.send_command('set player gnubg movefilter 4 0 0 8 0.160')
            self.gnubg_interface.send_command('set player gnubg movefilter 4 2 0 2 0.040')

            self.gnubg_interface.send_command('set player gnubg cube evaluation plies 2')
            self.gnubg_interface.send_command('set player gnubg cube evaluation prune on')
            self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.000')

        self.gnubg_interface.send_command('save setting')

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array', 'state_pixels'], print(mode)

        if mode == 'human':
            self.game.render()
            return True
        else:
            if self.viewer is None:
                self.viewer = Viewer(SCREEN_W, SCREEN_H)

            if mode == 'rgb_array':
                width = SCREEN_W
                height = SCREEN_H

            else:
                assert mode == 'state_pixels', print(mode)
                width = STATE_W
                height = STATE_H

            return self.viewer.render(board=self.game.board, bar=self.game.bar, off=self.game.off, state_w=width, state_h=height)


def evaluate_vs_gnubg(agent, env, n_episodes):
    wins = {WHITE: 0, BLACK: 0}

    for episode in range(n_episodes):
        observation, first_roll = env.reset()
        t = time.time()
        for i in count():
            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                env.gnubg = agent.roll_dice()
                env.update_game_board(env.gnubg.board)
                roll = env.gnubg.roll

            actions = env.get_valid_actions(roll)
            action = agent.choose_best_action(actions, env)

            observation_next, reward, done, info = env.step(action)
            # env.render(mode='rgb_array')

            if done:
                winner = WHITE if env.gnubg.winner == 'O' else BLACK
                wins[winner] += 1
                tot = wins[WHITE] + wins[BLACK]

                print("EVAL => Game={:<6} {:>15} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | gnubg={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(
                    episode + 1, '('+env.difficulty+')', info, env.gnubg.n_moves, agent.name, wins[WHITE], (wins[WHITE] / tot) * 100, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))
                break
            observation = observation_next

    env.gnubg_interface.send_command("new session")
    return wins
